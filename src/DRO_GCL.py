import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math


class GSCL(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(GSCL, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_csr = dataset.train_csr_record
        self.embedding_dim = self.flags_obj.embedding_dim
        self.cl_weight = self.flags_obj.cl_weight
        # self.num_community = self.flags_obj.num_community
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        self.Graph = dataset.symmetric_sub_graph
        self.DRO_Graph = dataset.DRO_symmetric_sub_graph

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

    @staticmethod
    def __dropout_x(x, static_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()

        random_index = torch.rand(len(values)) + static_prob
        random_index = random_index.int().bool()

        index = index[random_index]
        values = values[random_index] / static_prob

        graph = torch.sparse.FloatTensor(index.t(), values, size)

        return graph

    def __dropout(self, static_prob):
        # if self.flags_obj.adj_split:
        #     graph = []
        #     for g in self.origin_graph:
        #         graph.append(self.__dropout_x(g, static_prob))
        # else:
        graph = self.__dropout_x(self.Graph, static_prob)
        dro_graph = self.__dropout_x(self.DRO_Graph, static_prob)

        return graph, dro_graph

    def computer(self):
        user_embedding_list = [self.user_embedding]
        item_embedding_list = [self.item_embedding]

        dro_user_embedding_list = [self.user_embedding]
        dro_item_embedding_list = [self.item_embedding]

        if self.flags_obj.dropout:
            if self.training:
                graph, dro_graph = self.__dropout(self.flags_obj.static_prob)
            else:
                graph = self.Graph
                dro_graph = self.DRO_Graph
        else:
            graph = self.Graph
            dro_graph = self.DRO_Graph

        for layer in range(self.flags_obj.n_layers):
            user_embedding_list.append(graph @ item_embedding_list[layer])
            item_embedding_list.append(graph.T @ user_embedding_list[layer])

            dro_user_embedding_list.append(dro_graph @ item_embedding_list[layer])
            dro_item_embedding_list.append(dro_graph.T @ user_embedding_list[layer])

        user_embedding = sum(user_embedding_list) / (self.flags_obj.n_layers + 1)
        item_embedding = sum(item_embedding_list) / (self.flags_obj.n_layers + 1)
        dro_user_embedding = sum(dro_user_embedding_list) / (self.flags_obj.n_layers + 1)
        dro_item_embedding = sum(dro_item_embedding_list) / (self.flags_obj.n_layers + 1)

        return user_embedding, item_embedding, dro_user_embedding, dro_item_embedding

    def forward(self, users, positive_items, negative_items):
        user_embedding, item_embedding, dro_user_embedding, dro_item_embedding = self.computer()

        # cl loss
        temp = 0.2
        neg_score = torch.log(
            torch.exp(dro_user_embedding[users] @ user_embedding.T / temp).sum(1) + 1e-8).mean() + torch.log(
            torch.exp(dro_item_embedding[negative_items] @ item_embedding.T / temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((dro_user_embedding[users] * user_embedding[users]).sum(1) / temp, -5.0,
                                 5.0)).mean() + (
                        torch.clamp((dro_item_embedding[negative_items] * item_embedding[negative_items]).sum(1) / temp,
                                    -5.0, 5.0)).mean()
        loss_cl = -pos_score + neg_score

        # bpr loss
        users_embed = user_embedding[users]
        positive_embed = item_embedding[positive_items]
        negative_embed = item_embedding[negative_items]

        positive_scores = torch.sum(torch.mul(users_embed, positive_embed), dim=-1)
        negative_scores = torch.sum(torch.mul(users_embed, negative_embed), dim=-1)

        loss_bpr = torch.mean(func.softplus(negative_scores - positive_scores))

        loss = loss_bpr + self.cl_weight * loss_cl

        return loss
