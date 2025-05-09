import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math
import scipy.sparse as sp


def dCor(x, y):
    a = torch.norm(x[:, None] - x, p=2, dim=2)
    b = torch.norm(y[:, None] - y, p=2, dim=2)

    A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
    B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

    n = x.size(0)

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

    return dcor


def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class Structure(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(Structure, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        # self.popularity = dataset.popularity
        # self.activity = dataset.activity
        self.train_csr_record = dataset.train_csr_record
        self.csr_to_tensor = dataset.convert_sp_matrix_to_tensor
        self.embedding_dim = self.flags_obj.embedding_dim
        # self.num_community = self.flags_obj.num_community
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        # self.Graph = dataset.Graph
        # self.origin_Graph = dataset.origin_Graph
        # self.svd_u = dataset.svd_u
        # self.svd_v = dataset.svd_v
        self.Graph = dataset.SVD_origin_sub_graph
        self.SVD_Graph = dataset.SVD_symmetric_sub_graph
        self.origin_graph = dataset.origin_sub_graph

        self.U_mul_S = dataset.U_mul_S
        self.V_mul_S = dataset.V_mul_S

        self.user_map = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_map = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        # self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_map.data.uniform_(-stdv, stdv)
        self.item_map.data.uniform_(-stdv, stdv)

    def __dropout(self, static_prob):
        size = self.origin_graph.size()
        index = self.origin_graph.indices().t()
        values = self.origin_graph.values()

        random_index = torch.rand(len(values)) + static_prob
        random_index = random_index.int().bool()

        index = index[random_index]
        values = values[random_index] / static_prob

        graph = torch.sparse.FloatTensor(index.t(), values, size)

        return graph

    def computer(self, users, adjacent_items):
        # drop_tensor_graph = self.__dropout(self.flags_obj.static_prob)

        return self.user_map, self.item_map

    def forward(self):

        return self.user_map, self.item_map


class PGcn(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(PGcn, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.csr_to_tensor = dataset.convert_sp_matrix_to_tensor
        self.embedding_dim = self.flags_obj.embedding_dim
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        # self.Graph = dataset.symmetric_Graph
        self.Graph = dataset.symmetric_sub_graph
        self.origin_Graph = dataset.origin_Graph

        self.user_preference = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_preference = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_preference.data.uniform_(-stdv, stdv)
        self.item_preference.data.uniform_(-stdv, stdv)

    def __dropout(self, static_prob):
        size = self.Graph.size()
        index = self.Graph.indices().t()
        values = self.Graph.values()

        random_index = torch.rand(len(values)) + static_prob
        random_index = random_index.int().bool()

        index = index[random_index]
        values = values[random_index] / static_prob

        graph = torch.sparse.FloatTensor(index.t(), values, size)

        return graph

    def computer(self, users, adjacent_items):
        user_preference_inter_list = [self.user_preference]
        item_preference_inter_list = [self.item_preference]

        # if self.flags_obj.dropout:
        #     if self.training:
        #         graph_droped = self.__dropout(self.flags_obj.static_prob)
        #     else:
        #         graph_droped = self.Graph
        # else:
        #     graph_droped = self.Graph

        sample_csr = sp.csr_matrix((np.ones(users.shape[0]), (users.cpu(), adjacent_items.cpu())),
                                   shape=(self.num_users, self.num_items),
                                   dtype=np.int)
        # 采样过程user-positive item对会重复，导致正样本稀疏矩阵非零元素数值出现大于1的情况
        sample_csr = sample_csr.astype(np.bool).astype(np.int)

        sample_tensor = self.csr_to_tensor(sample_csr).coalesce().to(self.flags_obj.device)
        inter_graph_droped = self.Graph - self.Graph * sample_tensor
        # inter_graph_droped = graph_droped - graph_droped * sample_tensor

        for layer in range(self.flags_obj.n_layers):
            # 聚合邻居节点以及子环信息 todo: 没有自环信息？
            user_preference_inter_list.append(torch.sparse.mm(inter_graph_droped, item_preference_inter_list[layer]))
            item_preference_inter_list.append(torch.sparse.mm(inter_graph_droped.T, user_preference_inter_list[layer]))

            # preference = torch.sparse.mm(graph_droped, preference)
            # all_preference.append(preference)

        # inter_preference_user = sum(user_preference_inter_list)
        # inter_preference_item = sum(item_preference_inter_list)
        inter_preference_user = sum(user_preference_inter_list) / (self.flags_obj.n_layers + 1)
        inter_preference_item = sum(item_preference_inter_list) / (self.flags_obj.n_layers + 1)

        return inter_preference_user, inter_preference_item

    def forward(self):
        user_preference_list = [self.user_preference]
        item_preference_list = [self.item_preference]

        for layer in range(self.flags_obj.n_layers):
            # 聚合邻居节点以及子环信息 todo: 没有自环信息？
            user_preference_list.append(torch.sparse.mm(self.Graph, item_preference_list[layer]))
            item_preference_list.append(torch.sparse.mm(self.Graph.T, user_preference_list[layer]))

            # preference = torch.sparse.mm(graph_droped, preference)
            # all_preference.append(preference)
        # preference_user = sum(user_preference_list)
        # preference_item = sum(item_preference_list)
        preference_user = sum(user_preference_list) / (self.flags_obj.n_layers + 1)
        preference_item = sum(item_preference_list) / (self.flags_obj.n_layers + 1)

        return preference_user, preference_item


class MIA(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(MIA, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.popularity = dataset.popularity
        self.activity = dataset.activity
        self.embedding_dim = self.flags_obj.embedding_dim
        # self.num_community = self.flags_obj.num_community
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        # self.Graph = dataset.Graph
        # self.origin_Graph = dataset.origin_Graph
        self.pGcn = PGcn(dataset)
        self.structure = Structure(dataset)

        if self.flags_obj.discrepancy_loss == "L1":
            self.criterion_discrepancy = nn.L1Loss()
        elif self.flags_obj.discrepancy_loss == "L2":
            self.criterion_discrepancy = nn.MSELoss()
        elif self.flags_obj.discrepancy_loss == "dCor":
            self.criterion_discrepancy = dCor

        # self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_decision.data.uniform_(-stdv, stdv)
        self.item_decision.data.uniform_(-stdv, stdv)

    def structure_loss(self, users_structure, weak_items_structure, strong_items_structure):
        weak_structure_scores = torch.sum(torch.mul(users_structure, weak_items_structure), dim=-1)
        strong_structure_scores = torch.sum(torch.mul(users_structure, strong_items_structure), dim=-1)

        weak_strong_structure_loss = torch.mean(
            func.relu(torch.sub(weak_structure_scores, strong_structure_scores - 0.1)))
        # weak_strong_structure_loss = torch.mean(func.softplus(weak_structure_scores - strong_structure_scores))

        return weak_strong_structure_loss

    def fusion_sample(self, users, items_pool, items_weight, users_preference, items_preference, users_structure,
                      items_structure):
        users_embed = torch.cat((users_preference[users], users_structure[users]), dim=-1)
        update_weight = []
        for i in range(items_pool.shape[1]):
            items = items_pool[:, i]
            weight = items_weight[:, i]

            items_embed = torch.cat((items_preference[items], items_structure[items]), dim=-1)
            # items_rating = func.sigmoid(
            #     self.flags_obj.alpha * torch.sum(torch.mul(users_structure[users], items_structure[items]), dim=-1) + (
            #                 1 - self.flags_obj.alpha) * torch.sum(
            #         torch.mul(users_preference[users], items_preference[items]), dim=-1))

            # items_rating = 0.9 * func.sigmoid(
            #     torch.sum(torch.mul(users_structure[users], items_structure[items]), dim=-1)) + 0.1 * func.sigmoid(
            #     torch.sum(torch.mul(users_preference[users], items_preference[items]), dim=-1))

            # items_rating = func.sigmoid(torch.sum(torch.mul(users_structure[users], items_structure[items]), dim=-1))
            # 正辅助结构增强采样
            # items_rating = func.sigmoid(
            #     0.8 * torch.sum(torch.mul(users_structure[users], items_structure[items]), dim=-1) + 0.2 * torch.sum(
            #         torch.mul(items_structure[adjacent_items], items_structure[items]), dim=-1))
            # items_rating = func.sigmoid(0.8 * torch.sum(torch.mul(users_embed, items_embed), dim=-1) + 0.2 * torch.sum(
            #     torch.mul(adj_embed, items_embed), dim=-1))
            items_rating = func.sigmoid(torch.sum(torch.mul(users_embed, items_embed), dim=-1))
            # 融合采样
            weight = self.flags_obj.prior_weight * weight + items_rating
            update_weight.append(weight)

        update_weight = torch.stack(update_weight, dim=1)
        value, index = torch.max(update_weight, dim=1)
        negative_items = items_pool[torch.arange(0, index.shape[0]), index]

        return negative_items

    def random_sample(self, items_pool):
        index = torch.randint(0, items_pool.shape[1], (items_pool.shape[0],))
        negative_items = items_pool[torch.arange(0, items_pool.shape[0]), index]

        return negative_items

    def forward(self, users, adjacent_items, items_pool, items_weight):
        weak_items = items_pool[:, 0]
        weak_weight = items_weight[:, 0]
        strong_items = items_pool[:, -1]
        strong_weight = items_weight[:, -1]

        users_preference, items_preference = self.pGcn()
        users_preference_embed = users_preference[users]
        adj_preference_embed = items_preference[adjacent_items]

        users_structure, items_structure = self.structure()
        users_structure_embed = users_structure[users]
        adj_structure_embed = items_structure[adjacent_items]
        weak_structure_embed = items_structure[weak_items]
        strong_structure_embed = items_structure[strong_items]

        structure_loss = self.structure_loss(users_structure_embed, weak_structure_embed, strong_structure_embed)
        negative_items = self.fusion_sample(users, items_pool, items_weight, users_preference, items_preference,
                                            users_structure, items_structure)
        # negative_items = self.random_sample(items_pool)

        if self.flags_obj.str_intervention and self.flags_obj.pre_intervention:
            inter_users_structure, inter_items_structure = self.structure.computer(users, adjacent_items)
            inter_users_preference, inter_items_preference = self.pGcn.computer(users, adjacent_items)
            users_embed = torch.cat((inter_users_preference[users], inter_users_structure[users]), dim=-1)
            adj_embed = torch.cat((inter_items_preference[adjacent_items], inter_items_structure[adjacent_items]),
                                  dim=-1)
            negative_embed = torch.cat((inter_items_preference[negative_items], inter_items_structure[negative_items]),
                                       dim=-1)
        elif self.flags_obj.str_intervention:
            inter_users_structure, inter_items_structure = self.structure.computer(users, adjacent_items)
            users_embed = torch.cat((users_preference_embed, inter_users_structure[users]), dim=-1)
            adj_embed = torch.cat((adj_preference_embed, inter_items_structure[adjacent_items]), dim=-1)
            negative_embed = torch.cat((items_preference[negative_items], inter_items_structure[negative_items]),
                                       dim=-1)
        elif self.flags_obj.pre_intervention:
            inter_users_preference, inter_items_preference = self.pGcn.computer(users, adjacent_items)
            users_embed = torch.cat((inter_users_preference[users], users_structure_embed), dim=-1)
            adj_embed = torch.cat((inter_items_preference[adjacent_items], adj_structure_embed), dim=-1)
            negative_embed = torch.cat((inter_items_preference[negative_items], items_structure[negative_items]),
                                       dim=-1)
        else:
            users_embed = torch.cat((users_preference_embed, users_structure_embed), dim=-1)
            adj_embed = torch.cat((adj_preference_embed, adj_structure_embed), dim=-1)
            negative_embed = torch.cat((items_preference[negative_items], items_structure[negative_items]), dim=-1)

        adj_score = torch.sum(torch.mul(users_embed, adj_embed), dim=-1)
        negative_score = torch.sum(torch.mul(users_embed, negative_embed), dim=-1)
        bpr_loss = torch.mean(func.softplus(negative_score - adj_score))
        loss = bpr_loss + self.flags_obj.str_weight * structure_loss

        return loss
