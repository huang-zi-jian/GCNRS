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

        self.U_mul_S = dataset.U_mul_S
        self.V_mul_S = dataset.V_mul_S

        self.user_map = Parameter(torch.FloatTensor(self.flags_obj.q, self.embedding_dim))
        self.item_map = Parameter(torch.FloatTensor(self.flags_obj.q, self.embedding_dim))

        # self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_map.data.uniform_(-stdv, stdv)
        self.item_map.data.uniform_(-stdv, stdv)

    # def drop_graph_svd(self):
    def computer(self, users, adjacent_items):
        sample_scr = sp.csr_matrix((np.ones(users.shape[0]), (users.cpu(), adjacent_items.cpu())),
                                   shape=(self.num_users, self.num_items),
                                   dtype=np.int16)

        drop_csr_graph = self.train_csr_record - sample_scr
        drop_tensor_graph = self.csr_to_tensor(drop_csr_graph).coalesce().to(self.flags_obj.device)
        svd_u, s, svd_v = torch.svd_lowrank(drop_tensor_graph, q=self.flags_obj.q)
        u_mul_s = svd_u @ torch.diag(s)
        v_mul_s = svd_v @ torch.diag(s)

        interference_uses_structure = u_mul_s @ self.user_map
        interference_items_structure = v_mul_s @ self.item_map

        return interference_uses_structure, interference_items_structure

    def forward(self):
        users_structure = self.U_mul_S @ self.user_map
        items_structure = self.V_mul_S @ self.item_map

        return users_structure, items_structure


class Preference(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(Preference, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embedding_dim = self.flags_obj.embedding_dim

        self.user_preference = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_preference = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_preference.data.uniform_(-stdv, stdv)
        self.item_preference.data.uniform_(-stdv, stdv)

    def forward(self):

        return self.user_preference, self.item_preference


class PGcn(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(PGcn, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.popularity = dataset.popularity
        self.activity = dataset.activity
        self.embedding_dim = self.flags_obj.embedding_dim
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        self.Graph = dataset.symmetric_Graph
        self.origin_Graph = dataset.origin_Graph

        self.user_preference = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_preference = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))
        self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.embedding_dim)
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

    def forward(self):
        preference = torch.cat([self.user_preference, self.item_preference])
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
            all_preference.append(preference)

        preference_merge = torch.stack(all_preference, dim=-1)
        preference_merge = torch.sum(preference_merge, dim=-1)
        preference_user, preference_item = torch.split(preference_merge, [self.num_users, self.num_items], dim=0)

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
        self.sGcn = Structure(dataset)
        self.pGcn = PGcn(dataset)

        self.user_decision = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_decision = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        if self.flags_obj.discrepancy_loss == "L1":
            self.criterion_discrepancy = nn.L1Loss()
        elif self.flags_obj.discrepancy_loss == "L2":
            self.criterion_discrepancy = nn.MSELoss()
        elif self.flags_obj.discrepancy_loss == "dCor":
            self.criterion_discrepancy = dCor

        self.init_weight()

        # self.layer_norm = nn.LayerNorm(self.flags_obj.embedding_dim, elementwise_affine=False)

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_decision.data.uniform_(-stdv, stdv)
        self.item_decision.data.uniform_(-stdv, stdv)

    def preference_loss(self, users_decision, adj_items_decision):
        adj_decision_scores = torch.sum(torch.mul(users_decision, adj_items_decision), dim=-1)
        decision_loss = torch.mean(torch.log(1 / func.sigmoid(adj_decision_scores)))

        return decision_loss

    def structure_loss(self, users_structure, adj_items_structure, weak_items_structure, strong_items_structure):
        adj_structure_scores = torch.sum(torch.mul(users_structure, adj_items_structure), dim=-1)
        weak_structure_scores = torch.sum(torch.mul(users_structure, weak_items_structure), dim=-1)
        strong_structure_scores = torch.sum(torch.mul(users_structure, strong_items_structure), dim=-1)

        strong_adj_structure_loss = func.softplus(strong_structure_scores - adj_structure_scores)
        weak_strong_structure_loss = func.softplus(weak_structure_scores - strong_structure_scores)

        # weak_weight, strong_weight = torch.split(items_weight, [1, 1], dim=1)
        # weak_weight = weak_weight.squeeze()
        # strong_weight = strong_weight.squeeze()
        # alpha = 0.1
        # weight = alpha * torch.log(1 + strong_weight - weak_weight)
        # weight = strong_weight - weak_weight

        structure_loss = torch.mean((strong_adj_structure_loss + weak_strong_structure_loss) / 2)

        return structure_loss

    def forward(self, users, adjacent_items, items_pool, items_weight):
        weak_items, strong_items = torch.split(items_pool, [1, 1], dim=1)
        weak_items = weak_items.squeeze()
        strong_items = strong_items.squeeze()

        weak_weight, strong_weight = torch.split(items_weight, [1, 1], dim=1)
        weak_weight = weak_weight.squeeze()
        strong_weight = strong_weight.squeeze()

        # todo: temp越大，原始的三跳邻居数量weight的影响就越大?
        temp = 1
        weak_weight = func.sigmoid(torch.log((1 + weak_weight).pow(temp)))
        strong_weight = func.sigmoid(torch.log((1 + strong_weight).pow(temp)))

        users_preference, items_preference = self.pGcn()
        users_preference_embed = users_preference[users]
        adj_preference_embed = items_preference[adjacent_items]
        weak_preference_embed = items_preference[weak_items]
        strong_preference_embed = items_preference[strong_items]

        users_structure, items_structure = self.sGcn()
        users_structure_embed = users_structure[users]
        adj_structure_embed = items_structure[adjacent_items]
        weak_structure_embed = items_structure[weak_items]
        strong_structure_embed = items_structure[strong_items]

        structure_loss = self.structure_loss(users_structure_embed, adj_structure_embed, weak_structure_embed,
                                             strong_structure_embed)
        preference_loss = self.preference_loss(users_preference_embed, adj_preference_embed)

        inter_uses_structure, inter_items_structure = self.sGcn.computer(users, adjacent_items)

        user_exposure_embed = torch.cat((users_preference_embed, users_structure_embed), dim=-1)
        weak_exposure_embed = torch.cat((weak_preference_embed, weak_structure_embed), dim=-1)
        strong_exposure_embed = torch.cat((strong_preference_embed, strong_structure_embed), dim=-1)

        weak_exposure_rating = func.sigmoid(torch.sum(torch.mul(user_exposure_embed, weak_exposure_embed), dim=-1))
        strong_exposure_rating = func.sigmoid(torch.sum(torch.mul(user_exposure_embed, strong_exposure_embed), dim=-1))

        weak_weight = weak_weight + weak_exposure_rating
        strong_weight = strong_weight + strong_exposure_rating

        # 对strong_items重新排序
        update_strong_items = strong_items.clone()
        update_strong_items[strong_weight < weak_weight] = weak_items[strong_weight < weak_weight]
        strong_preference_embed = items_preference[update_strong_items]
        # strong_items_structure_embed = items_structure[update_strong_items]

        users_exposure_embed = torch.cat((users_preference_embed, inter_uses_structure[users]), dim=-1)
        adj_exposure_embed = torch.cat((adj_preference_embed, inter_items_structure[adjacent_items]), dim=-1)
        strong_exposure_embed = torch.cat((strong_preference_embed, inter_items_structure[update_strong_items]), dim=-1)

        # users_decision_embed = self.user_decision[users]
        # adj_items_decision_embed = self.item_decision[adjacent_items]
        # strong_items_decision_embed = self.item_decision[update_strong_items]
        users_decision_embed = torch.cat((users_preference_embed, self.user_decision[users]), dim=-1)
        adj_decision_embed = torch.cat((adj_preference_embed, self.item_decision[adjacent_items]), dim=-1)
        strong_decision_embed = torch.cat((strong_preference_embed, self.item_decision[update_strong_items]), dim=-1)

        decision_loss = self.preference_loss(users_decision_embed, adj_decision_embed)

        users_embed = torch.cat((users_exposure_embed, users_decision_embed), dim=-1)
        adj_embed = torch.cat((adj_exposure_embed, adj_decision_embed), dim=-1)
        strong_embed = torch.cat((strong_exposure_embed, strong_decision_embed), dim=-1)

        adj_score = torch.sum(torch.mul(users_embed, adj_embed), dim=-1)
        strong_score = torch.sum(torch.mul(users_embed, strong_embed), dim=-1)
        bpr_loss = torch.mean(func.softplus(strong_score - adj_score))

        loss = bpr_loss + decision_loss + 0.01 * structure_loss
        # todo: 结合FAWMF，增加GCN如何达到自适应效果？
        return loss
