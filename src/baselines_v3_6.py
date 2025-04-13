import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math
import scipy.sparse as sp


class LightGCN(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(LightGCN, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embedding_dim = self.flags_obj.embedding_dim
        self.Graph = dataset.symmetric_sub_graph

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))
        self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
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
        if self.flags_obj.adj_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, static_prob))
        else:
            graph = self.__dropout_x(self.Graph, static_prob)

        return graph

    def computer(self, users, positive_items):
        user_embed_inter_list = [self.user_embedding]
        item_embed_inter_list = [self.item_embedding]

        # if self.flags_obj.dropout:
        #     if self.training:
        #         graph_droped = self.__dropout(self.flags_obj.static_prob)
        #     else:
        #         graph_droped = self.Graph
        # else:
        #     graph_droped = self.Graph

        sample_csr = sp.csr_matrix((np.ones(users.shape[0]), (users.cpu(), positive_items.cpu())),
                                   shape=(self.num_users, self.num_items),
                                   dtype=np.int)
        # 采样过程user-positive item对会重复，导致正样本稀疏矩阵非零元素数值出现大于1的情况
        sample_csr = sample_csr.astype(np.bool).astype(np.int)

        sample_tensor = self.csr_to_tensor(sample_csr).coalesce().to(self.flags_obj.device)
        inter_graph_droped = self.Graph - self.Graph * sample_tensor
        # inter_graph_droped = graph_droped - graph_droped * sample_tensor

        for layer in range(self.flags_obj.n_layers):
            # 聚合邻居节点以及子环信息 todo: 没有自环信息？
            user_embed_inter_list.append(torch.sparse.mm(inter_graph_droped, item_embed_inter_list[layer]))
            item_embed_inter_list.append(torch.sparse.mm(inter_graph_droped.T, user_embed_inter_list[layer]))

        inter_embed_user = sum(user_embed_inter_list) / (self.flags_obj.n_layers + 1)
        inter_embed_item = sum(item_embed_inter_list) / (self.flags_obj.n_layers + 1)

        return inter_embed_user, inter_embed_item

    def forward(self):
        user_embed_list = [self.user_embedding]
        item_embed_list = [self.item_embedding]

        for layer in range(self.flags_obj.n_layers):
            # 聚合邻居节点以及子环信息 todo: 没有自环信息？
            user_embed_list.append(torch.sparse.mm(self.Graph, item_embed_list[layer]))
            item_embed_list.append(torch.sparse.mm(self.Graph.T, user_embed_list[layer]))

        embed_user = sum(user_embed_list) / (self.flags_obj.n_layers + 1)
        embed_item = sum(item_embed_list) / (self.flags_obj.n_layers + 1)

        return embed_user, embed_item


class LightGCL(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(LightGCL, self).__init__()
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
        self.SVD_Graph = dataset.SVD_symmetric_sub_graph

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
        svd_graph = self.__dropout_x(self.SVD_Graph, static_prob)

        return graph, svd_graph

    def computer(self):
        user_embedding_list = [self.user_embedding]
        item_embedding_list = [self.item_embedding]

        svd_user_embedding_list = [self.user_embedding]
        svd_item_embedding_list = [self.item_embedding]

        if self.flags_obj.dropout:
            if self.training:
                graph, svd_graph = self.__dropout(self.flags_obj.static_prob)
            else:
                graph = self.Graph
                svd_graph = self.SVD_Graph
        else:
            graph = self.Graph
            svd_graph = self.SVD_Graph

        for layer in range(self.flags_obj.n_layers):
            user_embedding_list.append(graph @ item_embedding_list[layer])
            item_embedding_list.append(graph.T @ user_embedding_list[layer])

            svd_user_embedding_list.append(svd_graph @ item_embedding_list[layer])
            svd_item_embedding_list.append(svd_graph.T @ user_embedding_list[layer])

        user_embedding = sum(user_embedding_list) / (self.flags_obj.n_layers + 1)
        item_embedding = sum(item_embedding_list) / (self.flags_obj.n_layers + 1)
        svd_user_embedding = sum(svd_user_embedding_list) / (self.flags_obj.n_layers + 1)
        svd_item_embedding = sum(svd_item_embedding_list) / (self.flags_obj.n_layers + 1)

        return user_embedding, item_embedding, svd_user_embedding, svd_item_embedding

    def forward(self, users, positive_items, negative_items):
        user_embedding, item_embedding, svd_user_embedding, svd_item_embedding = self.computer()

        # cl loss
        temp = 0.2
        neg_score = torch.log(
            torch.exp(svd_user_embedding[users] @ user_embedding.T / temp).sum(1) + 1e-8).mean() + torch.log(
            torch.exp(svd_item_embedding[negative_items] @ item_embedding.T / temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((svd_user_embedding[users] * user_embedding[users]).sum(1) / temp, -5.0,
                                 5.0)).mean() + (
                        torch.clamp((svd_item_embedding[negative_items] * item_embedding[negative_items]).sum(1) / temp,
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


class FAWMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(FAWMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embedding_dim = self.flags_obj.embedding_dim
        self.num_community = 20
        self.Graph = dataset.origin_Graph

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        self.theta_user = Parameter(torch.FloatTensor(self.num_users, self.num_community))
        self.w1 = Parameter(torch.FloatTensor(self.num_items, 1))
        self.w2 = Parameter(torch.FloatTensor(self.num_items, 1))
        self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)
        # self.user_embedding.data.uniform_(0, 0.1)
        # self.item_embedding.data.uniform_(0, 0.1)
        stdv = 1. / math.sqrt(self.num_community)
        self.theta_user.data.uniform_(-stdv, stdv)
        self.w1.data.uniform_(-0.2, 0.2)
        self.w2.data.uniform_(-0.2, 0.2)

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

        return graph

    def computer(self):
        # theta_user = func.normalize(func.relu(self.theta_user), dim=-1, p=1)
        theta_user = func.softmax(self.theta_user, dim=-1)
        temp_pack = torch.zeros(size=(self.num_items, self.num_community), device=self.flags_obj.device)
        # theta_item = func.softmax(self.theta_item, dim=-1)

        all_theta = torch.cat([theta_user, temp_pack])

        if self.flags_obj.dropout:
            if self.training:
                graph_droped = self.__dropout(self.flags_obj.static_prob)
            else:
                graph_droped = self.Graph
        else:
            graph_droped = self.Graph

        # 聚合邻居节点以及子环信息 todo: 没有自环信息？
        z = torch.sparse.mm(graph_droped, all_theta)

        _, z1 = torch.split(z, [self.num_users, self.num_items], dim=0)
        z1 = func.sigmoid(z1 * self.w1 + self.w2)

        return theta_user, z1

    def getEmbedding(self, users, positive_items, negative_items):
        users_embed_origin = self.user_embedding[users]
        positive_embed_origin = self.item_embedding[positive_items]
        negative_embed_origin = self.item_embedding[negative_items]

        return users_embed_origin, positive_embed_origin, negative_embed_origin

    def forward(self, users, positive_items, negative_items):
        users_embed_origin, positive_embed_origin, negative_embed_origin = self.getEmbedding(users.long(),
                                                                                             positive_items.long(),
                                                                                             negative_items.long())
        regular_loss1 = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                   positive_embed_origin.norm(2).pow(2) +
                                   negative_embed_origin.norm(2).pow(2)) / float(len(users))
        regular_loss2 = (1 / 2) * (self.w1.norm(2).pow(2) + self.w2.norm(2).pow(2)) / float(self.num_items)
        regular_loss3 = (1 / 2) * (self.theta_user.norm(2).pow(2)) / float(self.num_users)

        regular_loss = self.flags_obj.weight_decay * (regular_loss1 + regular_loss3 + 10 * regular_loss2)

        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)

        # theta_, z1 = self.exposure_weight()
        theta_user, z1 = self.computer()

        gamma_positive = torch.mul(theta_user[users], z1[positive_items])
        gamma_positive = torch.sum(gamma_positive, dim=-1)
        gamma_negative = torch.mul(theta_user[users], z1[negative_items])
        gamma_negative = torch.sum(gamma_negative, dim=-1)

        positive_label = torch.ones(size=(gamma_positive.shape[0],), device=self.flags_obj.device)
        negative_label = torch.zeros(size=(gamma_negative.shape[0],), device=self.flags_obj.device)

        # FAWMF损失函数
        positive_rating = func.sigmoid(positive_scores)
        negative_rating = func.sigmoid(negative_scores)
        rating = torch.stack((positive_rating, negative_rating), dim=0)
        label = torch.stack((positive_label, negative_label), dim=0)
        gamma = torch.stack((gamma_positive, gamma_negative), dim=0)

        bce_loss = self.BCE(rating, label)
        bce_loss = torch.mul(gamma, bce_loss)
        mean_bce_loss = torch.mean(bce_loss)

        gamma_positive_kl = torch.mean(torch.log2(1 / gamma_positive))

        loss = mean_bce_loss + 0.01 * gamma_positive_kl + regular_loss

        # mf_loss = torch.pow(rating - label, 2)
        # wmf_loss = torch.mul(gamma, mf_loss)
        # wmf_loss_mean = torch.mean(wmf_loss)
        #
        # unknown_loss = torch.pow(1e-5 - label, 2)
        # unknown_loss = torch.mul(1 - gamma, unknown_loss)
        # unknown_loss_mean = torch.mean(unknown_loss)
        #
        # loss = wmf_loss_mean + 0.1 * unknown_loss_mean + regular_loss

        # loss = mean_mf_loss + 0.1 * (mean_unknown_loss - mean_gamma_uncertain_loss) + regular_loss

        return loss


class BPR(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(BPR, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        # self.user_embedding.data.uniform_(0, 0.1)
        # self.item_embedding.data.uniform_(0, 0.1)
        stdv = 1. / math.sqrt(self.flags_obj.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

    def getEmbedding(self, users, positive_items, negative_items):
        users_embed_origin = self.user_embedding[users]
        positive_embed_origin = self.item_embedding[positive_items]
        negative_embed_origin = self.item_embedding[negative_items]

        return users_embed_origin, positive_embed_origin, negative_embed_origin

    def forward(self, users, positive_items, negative_items):
        users_embed_origin, positive_embed_origin, negative_embed_origin = self.getEmbedding(users.long(),
                                                                                             positive_items.long(),
                                                                                             negative_items.long())
        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        # positive_rating = func.sigmoid(positive_scores)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)
        # negative_rating = func.sigmoid(negative_scores)

        mean_loss = torch.mean(func.softplus(negative_scores - positive_scores))
        # mean_loss = torch.mean(- func.logsigmoid(positive_scores - negative_scores))
        # loss = loss + self.flags_obj.weight_decay * regular_loss

        return mean_loss


class WMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(WMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        # self.user_embedding.data.uniform_(0, 0.1)
        # self.item_embedding.data.uniform_(0, 0.1)
        stdv = 1. / math.sqrt(self.flags_obj.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

    def getEmbedding(self, users, positive_items, negative_items):
        users_embed_origin = self.user_embedding[users]
        positive_embed_origin = self.item_embedding[positive_items]
        negative_embed_origin = self.item_embedding[negative_items]

        return users_embed_origin, positive_embed_origin, negative_embed_origin

    def forward(self, users, positive_items, negative_items):
        users_embed_origin, positive_embed_origin, negative_embed_origin = self.getEmbedding(users.long(),
                                                                                             positive_items.long(),
                                                                                             negative_items.long())
        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        positive_rating = func.sigmoid(positive_scores)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)
        negative_rating = func.sigmoid(negative_scores)

        positive_label = torch.ones(size=(positive_rating.shape[0],), device=self.flags_obj.device)
        negative_label = torch.zeros(size=(negative_rating.shape[0],), device=self.flags_obj.device)

        rating = torch.stack((positive_rating, negative_rating), dim=0)
        label = torch.stack((positive_label, negative_label), dim=0)

        weight = 1 + 1 * torch.log(1 + label)

        mf_loss = torch.pow(rating - label, 2)
        wmf_loss = torch.mul(weight, mf_loss)
        mean_wmf_loss = torch.mean(wmf_loss)

        # bpr_score = func.softplus(negative_scores - positive_scores)
        # loss = torch.mean(bpr_score)

        return mean_wmf_loss


# EXMFgamma参数过多，num_users * num_items，内存不够
class EXMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(EXMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))
        self.gamma = Parameter(torch.randn([self.num_users, self.num_items], dtype=torch.float32))

        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        # self.user_embedding.data.uniform_(0, 0.1)
        # self.item_embedding.data.uniform_(0, 0.1)
        stdv = 1. / math.sqrt(self.flags_obj.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

    def getEmbedding(self, users, positive_items, negative_items):
        users_embed_origin = self.user_embedding[users]
        positive_embed_origin = self.item_embedding[positive_items]
        negative_embed_origin = self.item_embedding[negative_items]

        return users_embed_origin, positive_embed_origin, negative_embed_origin

    def forward(self, users, positive_items, negative_items):
        users = users.long()
        positive_items = positive_items.long()
        negative_items = negative_items.long()
        users_embed_origin, positive_embed_origin, negative_embed_origin = self.getEmbedding(users, positive_items,
                                                                                             negative_items)

        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        positive_rating = func.sigmoid(positive_scores)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)
        negative_rating = func.sigmoid(negative_scores)

        positive_label = torch.ones(size=(positive_rating.shape[0],), device=self.flags_obj.device)
        negative_label = torch.zeros(size=(negative_rating.shape[0],), device=self.flags_obj.device)

        rating = torch.stack((positive_rating, negative_rating), dim=0)
        label = torch.stack((positive_label, negative_label), dim=0)
        positive_gamma = func.sigmoid(self.gamma[users, positive_items])
        negative_gamma = func.sigmoid(self.gamma[users, negative_items])
        gamma = torch.stack((positive_gamma, negative_gamma), dim=0)

        mf_loss = torch.pow(rating - label, 2)
        wmf_loss = torch.mul(gamma, mf_loss)
        wmf_loss_mean = torch.mean(wmf_loss)

        unknown_loss = torch.pow(1e-5 - label, 2)
        unknown_loss = torch.mul(1 - gamma, unknown_loss)
        unknown_loss_mean = torch.mean(unknown_loss)
        loss = wmf_loss_mean + 0.1 * unknown_loss_mean

        return loss
