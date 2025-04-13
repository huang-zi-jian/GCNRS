import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math


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


class Structure(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(Structure, self).__init__()
        self.flags_obj = dataset.flags_obj
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_community = self.flags_obj.num_community

        self.user_structure = Parameter(torch.FloatTensor(self.num_users, self.num_community))
        self.item_structure = Parameter(torch.FloatTensor(self.num_items, self.num_community))

        self.distance = nn.PairwiseDistance(p=2)
        self.cos_similar = nn.CosineSimilarity()
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.num_community)
        self.user_structure.data.uniform_(-stdv, stdv)
        self.item_structure.data.uniform_(-stdv, stdv)

    def forward(self, users, adjacent_items, weak_items, strong_items):
        users_structure = self.user_structure[users.long()]
        adjacent_items_structure = self.item_structure[adjacent_items.long()]
        weak_items_structure = self.item_structure[weak_items.long()]
        strong_items_structure = self.item_structure[strong_items.long()]

        users_map = func.normalize(users_structure, p=2, dim=-1)
        adjacent_items_map = func.normalize(adjacent_items_structure, p=2, dim=-1)
        weak_items_map = func.normalize(weak_items_structure, p=2, dim=-1)
        strong_items_map = func.normalize(strong_items_structure, p=2, dim=-1)

        return users_structure, adjacent_items_structure, weak_items_structure, strong_items_structure
        # cos_similar = self.cos_similar(users_map, adjacent_items_map)
        # pairwise_distance = self.distance(users_map, adjacent_items_map) / 2

        # gamma_structure = {}
        # adjacent_gamma_structure = (2 - self.distance(users_map, adjacent_items_map)) / 2
        # weak_gamma_structure = (2 - self.distance(users_map, weak_items_map)) / 2
        # strong_gamma_structure = (2 - self.distance(users_map, strong_items_map)) / 2
        # gamma_structure['adjacent'] = adjacent_gamma_structure
        # gamma_structure['weak'] = weak_gamma_structure
        # gamma_structure['strong'] = strong_gamma_structure
        #
        # return gamma_structure


class PGcn(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(PGcn, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.popularity = dataset.popularity
        self.activity = dataset.activity
        self.embedding_dim = self.flags_obj.embedding_dim
        self.num_community = self.flags_obj.num_community
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        self.Graph = dataset.symmetric_Graph
        self.origin_Graph = dataset.origin_Graph

        # self.user_preference = Parameter(torch.FloatTensor(self.num_users, self.num_community))
        # self.item_preference = Parameter(torch.FloatTensor(self.num_items, self.num_community))
        self.user_preference = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_preference = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))
        self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.num_community)
        self.user_preference.data.uniform_(-stdv, stdv)
        self.item_preference.data.uniform_(-stdv, stdv)

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
        preference = torch.cat([self.user_preference, self.item_preference])
        # preference = func.normalize(func.leaky_relu(preference, negative_slope=0.1), p=2, dim=-1)
        # preference = func.leaky_relu(preference, negative_slope=0.1)
        all_preference = [preference]

        if self.flags_obj.dropout:
            if self.training:
                graph_droped = self.__dropout(self.flags_obj.static_prob)
            else:
                graph_droped = self.Graph
        else:
            graph_droped = self.Graph

        for layer in range(self.flags_obj.n_layers):
            # 聚合邻居节点以及子环信息 todo: 没有自环信息？
            preference = torch.sparse.mm(graph_droped, preference)
            # preference = func.normalize(func.leaky_relu(preference, negative_slope=0.1), p=2, dim=-1)
            # preference = func.leaky_relu(preference, negative_slope=0.1)
            all_preference.append(preference)

        preference_merge = torch.stack(all_preference, dim=-1)
        preference_merge = torch.mean(preference_merge, dim=-1)
        lgn_preference_user, lgn_preference_item = torch.split(preference_merge, [self.num_users, self.num_items],
                                                               dim=0)

        return lgn_preference_user, lgn_preference_item

    # def get_multi_hop_vector(self, embedding, items):
    #     zero_embedding = torch.zeros(embedding.shape[-1], device=self.flags_obj.device)
    #     multi_hop_embedding = embedding[items.long()]
    #     multi_hop_embedding[items == -1] = zero_embedding
    #
    #     return multi_hop_embedding

    def forward(self, users, adjacent_items, weak_items, strong_items):
        users_preference, items_preference = self.computer()

        users_pre_embed = users_preference[users]
        adj_items_pre_embed = items_preference[adjacent_items]
        weak_items_pre_embed = items_preference[weak_items]
        strong_items_pre_embed = items_preference[strong_items]

        # gamma_preference = {}
        # adjacent_gamma_preference = torch.sum(torch.mul(users_pre_embed, adj_items_pre_embed), dim=-1)
        # weak_gamma_preference = torch.sum(torch.mul(users_pre_embed, weak_items_pre_embed), dim=-1)
        # strong_gamma_preference = torch.sum(torch.mul(users_pre_embed, strong_items_pre_embed), dim=-1)
        #
        # gamma_preference['adjacent'] = func.sigmoid(adjacent_gamma_preference)
        # gamma_preference['weak'] = func.sigmoid(weak_gamma_preference)
        # gamma_preference['strong'] = func.sigmoid(strong_gamma_preference)

        return users_pre_embed, adj_items_pre_embed, weak_items_pre_embed, strong_items_pre_embed


class MIA(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(MIA, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.popularity = dataset.popularity
        self.activity = dataset.activity
        self.embedding_dim = self.flags_obj.embedding_dim
        self.num_community = self.flags_obj.num_community
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        self.Graph = dataset.symmetric_Graph
        self.origin_Graph = dataset.origin_Graph

        # self.structure = Structure(dataset)
        self.pGcn = PGcn(dataset)

        self.BCE = torch.nn.BCELoss(reduction='none')
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.distance = nn.PairwiseDistance(p=2)
        # self.Hinge = torch.nn.HingeEmbeddingLoss(margin=1)

        if self.flags_obj.discrepancy_loss == "L1":
            self.criterion_discrepancy = nn.L1Loss()
        elif self.flags_obj.discrepancy_loss == "L2":
            self.criterion_discrepancy = nn.MSELoss()
        elif self.flags_obj.discrepancy_loss == "dCor":
            self.criterion_discrepancy = dCor

        # self.init_weight()

        # self.layer_norm = nn.LayerNorm(self.flags_obj.embedding_dim, elementwise_affine=False)

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

    @staticmethod
    def preference_loss(users_preference, adj_items_preference):
        adj_preference = torch.sum(torch.mul(users_preference, adj_items_preference), dim=-1)
        preference_loss = torch.mean(torch.log(1 / func.sigmoid(adj_preference)))

        return preference_loss

    def structure_loss(self, users_structure, adj_items_structure, weak_items_structure, strong_items_structure,
                       items_weight):
        adjacent_structure = (2 - self.distance(users_structure, adj_items_structure)) / 2
        weak_structure = (2 - self.distance(users_structure, weak_items_structure)) / 2
        strong_structure = (2 - self.distance(users_structure, strong_items_structure)) / 2

        weak_weight, strong_weight = torch.split(items_weight, [1, 1], dim=1)
        weak_weight = weak_weight.squeeze()
        strong_weight = strong_weight.squeeze()
        alpha = 0.1
        weight = alpha * torch.log(1 + strong_weight - weak_weight)

        # hinge_margin = 0.1
        # strong_adj_structure_loss = func.relu(torch.sub(strong_structure, adjacent_structure - hinge_margin))
        # weak_strong_structure_loss = func.relu(torch.sub(weak_structure, strong_structure - hinge_margin))
        strong_adj_structure_loss = func.softplus(strong_structure - adjacent_structure)
        weak_strong_structure_loss = func.softplus(weak_structure - strong_structure)

        # loss = weight * gamma_structure_loss2
        structure_loss = torch.mean((strong_adj_structure_loss + weight * weak_strong_structure_loss) / 2)

        return structure_loss

    def discrepancy_loss(self, users, adjacent_items, weak_items, strong_items):
        users_structure_embed = self.structure.user_structure[users]
        adj_structure_embed = self.structure.item_structure[adjacent_items]
        weak_structure_embed = self.structure.item_structure[weak_items]
        strong_structure_embed = self.structure.item_structure[strong_items]

        users_preference_embed = self.pGcn.user_preference[users]
        adj_preference_embed = self.pGcn.item_preference[adjacent_items]
        weak_preference_embed = self.pGcn.item_preference[weak_items]
        strong_preference_embed = self.pGcn.item_preference[strong_items]

        discrepancy_loss = - (self.criterion_discrepancy(users_structure_embed, users_preference_embed) +
                              self.criterion_discrepancy(adj_structure_embed, adj_preference_embed) +
                              self.criterion_discrepancy(weak_structure_embed, weak_preference_embed) +
                              self.criterion_discrepancy(strong_structure_embed, strong_preference_embed))

        return discrepancy_loss

    def click_pre(self, users):
        users_preference, items_preference = self.pGcn.computer()
        users_embed = users_preference[users.long()]
        items_embed = items_preference
        rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))

        return rating

    def forward(self, users, adjacent_items, items_pool, items_weight):
        # users_embed_origin = self.user_embedding[users.long()]
        # adjacent_embed_origin = self.item_embedding[adjacent_items.long()]
        weak_items, strong_items = torch.split(items_pool, [1, 1], dim=1)
        weak_items = weak_items.squeeze()
        strong_items = strong_items.squeeze()
        # weak_embed_origin = self.item_embedding[weak_items.long()]
        # strong_embed_origin = self.item_embedding[strong_items.long()]

        # users_structure, adj_items_structure, weak_items_structure, strong_items_structure = self.structure(users,
        #                                                                                                     adjacent_items,
        #                                                                                                     weak_items,
        #                                                                                                     strong_items)
        users_preference, adj_items_preference, weak_items_preference, strong_items_preference = self.pGcn(users,
                                                                                                           adjacent_items,
                                                                                                           weak_items,
                                                                                                           strong_items)

        # structure_loss = self.structure_loss(users_structure, adj_items_structure, weak_items_structure,
        #                                      strong_items_structure, items_weight)
        # preference_loss = self.preference_loss(users_preference, adj_items_preference)

        # users_embed = torch.cat((users_preference, users_structure), dim=-1)
        # adj_items_embed = torch.cat((adj_items_preference, adj_items_structure), dim=-1)
        # weak_items_embed = torch.cat((weak_items_preference, weak_items_structure), dim=-1)
        # strong_items_embed = torch.cat((strong_items_preference, strong_items_structure), dim=-1)

        # adjacent_scores = torch.sum(torch.mul(users_embed, adj_items_embed), dim=-1)
        # weak_scores = torch.sum(torch.mul(users_embed, weak_items_embed), dim=-1)
        # strong_scores = torch.sum(torch.mul(users_embed, strong_items_embed), dim=-1)
        # adjacent_rating = func.sigmoid(adjacent_scores)
        # weak_rating = func.sigmoid(weak_scores)
        # strong_rating = func.sigmoid(strong_scores)

        adjacent_scores = torch.sum(torch.mul(users_preference, adj_items_preference), dim=-1)
        weak_scores = torch.sum(torch.mul(users_preference, weak_items_preference), dim=-1)
        strong_scores = torch.sum(torch.mul(users_preference, strong_items_preference), dim=-1)
        adjacent_rating = func.sigmoid(adjacent_scores)
        weak_rating = func.sigmoid(weak_scores)
        strong_rating = func.sigmoid(strong_scores)

        positive_label = torch.ones(size=(adjacent_items.shape[0],), device=self.flags_obj.device)
        negative_label = torch.zeros(size=(adjacent_items.shape[0],), device=self.flags_obj.device)

        rating = torch.stack((adjacent_rating, weak_rating, strong_rating), dim=0)
        label = torch.stack((positive_label, negative_label, negative_label), dim=0)

        # gamma = torch.stack((adjacent_gamma, weak_gamma, strong_gamma), dim=0)

        # weak_loss = func.softplus(weak_scores - adjacent_scores)
        # strong_loss = func.softplus(strong_scores - adjacent_scores)
        # weight_weak_loss = torch.mul((weak_gamma + adjacent_gamma) / 2, weak_loss)
        # weight_strong_loss = torch.mul((strong_loss + adjacent_gamma) / 2, strong_loss)
        # mean_soft_loss = torch.mean((weight_weak_loss + weight_strong_loss))

        hinge_margin = 0.1
        rating_adj_weak_hinge_loss = func.relu(torch.sub(weak_rating, adjacent_rating - hinge_margin))
        rating_adj_strong_hinge_loss = func.relu(torch.sub(strong_rating, adjacent_rating - hinge_margin))
        rating_mean_hinge_loss = torch.mean(rating_adj_weak_hinge_loss + rating_adj_strong_hinge_loss)

        # score_adj_weak_hinge_loss = func.relu(torch.sub(weak_scores, adjacent_scores - hinge_margin))
        # score_adj_strong_hinge_loss = func.relu(torch.sub(strong_scores, adjacent_scores - hinge_margin))
        # score_weight_weak_hinge_loss = torch.mul((adjacent_gamma + weak_gamma) / 2, score_adj_weak_hinge_loss)
        # score_weight_strong_hinge_loss = torch.mul((adjacent_gamma + strong_gamma) / 2, score_adj_strong_hinge_loss)
        # score_mean_hinge_loss = torch.mean(score_weight_weak_hinge_loss + score_weight_strong_hinge_loss)

        bce_loss = self.BCE(rating, label)
        # weight_bce_loss = torch.mul(gamma, bce_loss)
        mean_bce_loss = torch.mean(bce_loss)

        # discrepancy = - (self.criterion_discrepancy(users_structure, users_preference) +
        #                  self.criterion_discrepancy(adj_items_structure, adj_items_preference) +
        #                  self.criterion_discrepancy(weak_items_structure, weak_items_preference) +
        #                  self.criterion_discrepancy(strong_items_structure, strong_items_preference))

        # discrepancy = self.discrepancy_loss(users, adjacent_items, weak_items, strong_items)

        # loss = mean_bce_loss + 0.01 * (preference_loss + structure_loss) + 0.01 * discrepancy

        mean_loss = torch.mean(func.softplus(strong_scores - adjacent_scores))

        # loss = mean_bce_loss
        # todo: 结合FAWMF，增加GCN如何达到自适应效果？
        return mean_loss
