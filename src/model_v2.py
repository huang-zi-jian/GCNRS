import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math


class TransH(nn.Module):
    def __init__(self, flags_obj):
        super(TransH, self).__init__()
        self.num_community = flags_obj.num_community

        self.user_structure = Parameter(torch.FloatTensor(self.num_users, self.num_community))
        self.item_structure = Parameter(torch.FloatTensor(self.num_items, self.num_community))

        self.relation_embedding = Parameter(torch.FloatTensor(1, self.num_community))
        self.relationHyper = Parameter(torch.FloatTensor(1, self.num_community))

        self.distancmodel_v2.pye_func = nn.PairwiseDistance(p=2)
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.num_community)
        self.user_structure.data.uniform_(-stdv, stdv)
        self.item_structure.data.uniform_(-stdv, stdv)

    def distance(self, user, item):
        user_s = self.user_structure[user]
        item_s = self.item_structure[item]
        relation = self.relation_embedding[0]

        relation_hyper = func.normalize(self.relationHyper[0], p=2, dim=-1)

        user_map = user_s - relation_hyper * torch.sum(user_s * relation_hyper, dim=1)
        item_map = item_s - relation_hyper * torch.sum(item_s * relation_hyper, dim=1)

        distance_score = self.distance_func(user_map + relation, item_map)

        return distance_score


class LightGCN(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(LightGCN, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.Graph = dataset.Graph

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.flags_obj.embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.flags_obj.embedding_dim)

    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

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

    def computer(self):
        users_embed = self.user_embedding.weight
        items_embed = self.item_embedding.weight

        all_embeds = torch.cat([users_embed, items_embed])
        embeds = [all_embeds]

        if self.flags_obj.dropout:
            if self.training:
                graph_droped = self.__dropout(self.flags_obj.static_prob)
            else:
                graph_droped = self.Graph
        else:
            graph_droped = self.Graph

        for layer in range(self.flags_obj.n_layers):
            if self.flags_obj.adj_split:
                temp_embed = []
                for i in range(len(graph_droped)):
                    temp_embed.append(torch.sparse.mm(graph_droped[i], embeds))

                side_embed = torch.cat(temp_embed, dim=0)
                all_embeds = side_embed
            else:
                # 聚合邻居节点以及子环信息 todo: 没有自环信息？
                all_embeds = torch.sparse.mm(graph_droped, all_embeds)

            embeds.append(all_embeds)
        embeds = torch.stack(embeds, dim=1)
        light_out = torch.mean(embeds, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items], dim=0)

        return users, items

    def getEmbedding(self, users, positive_items, negative_items):
        all_users, all_items = self.computer()
        users_embed = all_users[users]
        positive_embed = all_items[positive_items]
        negative_embed = all_items[negative_items]
        users_embed_origin = self.user_embedding(users)
        positive_embed_origin = self.item_embedding(positive_items)
        negative_embed_origin = self.item_embedding(negative_items)

        return users_embed, positive_embed, negative_embed, users_embed_origin, positive_embed_origin, negative_embed_origin

    def forward(self, users, positive_items, negative_items):
        users_embed, positive_embed, negative_embed, users_embed_origin, positive_embed_origin, negative_embed_origin = self.getEmbedding(
            users.long(), positive_items.long(), negative_items.long())
        regular_loss = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                  positive_embed_origin.norm(2).pow(2) +
                                  negative_embed_origin.norm(2).pow(2)) / float(len(users))
        positive_scores = torch.mul(users_embed, positive_embed)
        positive_scores = torch.sum(positive_scores, dim=1)
        negative_scores = torch.mul(users_embed, negative_embed)
        negative_scores = torch.sum(negative_scores, dim=1)

        loss = torch.mean(func.softplus(negative_scores - positive_scores))
        loss = loss + self.flags_obj.weight_decay * regular_loss

        return loss


class FAWMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(FAWMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_community = self.flags_obj.num_community
        self.Graph = dataset.origin_Graph

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))

        self.theta_user = Parameter(torch.FloatTensor(self.num_users, self.num_community))
        self.w1 = Parameter(torch.FloatTensor(self.num_items, 1))
        self.w2 = Parameter(torch.FloatTensor(self.num_items, 1))
        self.BCE = torch.nn.BCELoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.flags_obj.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)
        # self.user_embedding.data.uniform_(0, 0.1)
        # self.item_embedding.data.uniform_(0, 0.1)
        self.theta_user.data.uniform_(-0.5, 0.5)
        self.w1.data.uniform_(0, 0.1)
        self.w2.data.uniform_(-0.1, 0.1)

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

        regular_loss = self.flags_obj.weight_decay * (regular_loss1 + regular_loss3) + 0.1 * regular_loss2

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
        mf_loss = torch.mul(gamma, bce_loss)
        mean_mf_loss = torch.mean(mf_loss)

        epsilon = torch.full([label.shape[0], label.shape[1]], 1e-3, device=self.flags_obj.device)
        unknown_loss = self.BCE(epsilon, label)
        unknown_loss = torch.mul(1 - gamma, unknown_loss)
        mean_unknown_loss = torch.mean(unknown_loss)

        gamma_uncertain_loss = self.BCE(gamma, gamma)
        mean_gamma_uncertain_loss = torch.mean(gamma_uncertain_loss)

        loss = mean_mf_loss + 0.1 * (mean_unknown_loss - mean_gamma_uncertain_loss) + regular_loss

        return loss


class MultiFAWMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(MultiFAWMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.popularity = dataset.popularity
        self.activity = dataset.activity
        self.num_community = self.flags_obj.num_community
        # todo：修改聚合权重，
        #  1、流行度平方根的倒数
        #  2、将上述邻居节点的倒数进行归一化
        #  3、减小正则化系数
        self.Graph = dataset.origin_Graph

        self.transH = TransH(self.flags_obj)
        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))

        self.theta_user = Parameter(torch.FloatTensor(self.num_users, self.num_community))
        self.theta_item = Parameter(torch.FloatTensor(self.num_items, self.num_community))
        self.w1_item = Parameter(torch.FloatTensor(self.num_items, 1))
        self.w2_item = Parameter(torch.FloatTensor(self.num_items, 1))
        self.w1_user = Parameter(torch.FloatTensor(self.num_users, 1))
        self.w2_user = Parameter(torch.FloatTensor(self.num_users, 1))
        self.BCE = torch.nn.BCELoss(reduction='none')
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.init_weight()

        # self.layer_norm = nn.LayerNorm(self.flags_obj.embedding_dim, elementwise_affine=False)

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.flags_obj.embedding_dim)
        self.user_embedding.data.uniform_(-stdv, stdv)
        self.item_embedding.data.uniform_(-stdv, stdv)

        self.theta_user.data.uniform_(-0.5, 0.5)
        self.theta_item.data.uniform_(-0.5, 0.5)

        self.w1_user.data.uniform_(0, 0.1)
        self.w2_user.data.uniform_(-0.1, 0.1)
        self.w1_item.data.uniform_(0, 0.1)
        self.w2_item.data.uniform_(-0.1, 0.1)

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

    def computer(self, users, adjacent_items, intermediate_items, distant_items):
        theta_user = func.softmax(self.theta_user, dim=-1)
        theta_item = func.softmax(self.theta_item, dim=-1)
        theta = torch.cat([theta_user, theta_item])
        all_theta = [theta]
        all_z = [(theta_user, theta_item)]

        if self.flags_obj.dropout:
            if self.training:
                graph_droped = self.__dropout(self.flags_obj.static_prob)
            else:
                graph_droped = self.Graph
        else:
            graph_droped = self.Graph

        for layer in range(self.flags_obj.n_layers):
            # 聚合邻居节点以及子环信息 todo: 没有自环信息？
            theta = torch.sparse.mm(graph_droped, theta)
            all_theta.append(theta)

            z_user, z_item = torch.split(theta, [self.num_users, self.num_items], dim=0)
            # z_user = func.leaky_relu(z_user * self.w1_user + self.w2_user)
            # z_item = func.leaky_relu(z_item * self.w1_item + self.w2_item)
            z_user = func.sigmoid(z_user * self.w1_user + self.w2_user)
            z_item = func.sigmoid(z_item * self.w1_item + self.w2_item)
            all_z.append((z_user, z_item))

        theta_merge = torch.stack(all_theta, dim=-1)
        theta_merge = torch.mean(theta_merge, dim=-1)
        lgc_theta_user, lgc_theta_item = torch.split(theta_merge, [self.num_users, self.num_items], dim=0)

        def gamma_genarate():
            gamma_feature = {}
            adjacent_gamma_feature = torch.sum(torch.mul(lgc_theta_user[users], lgc_theta_item[adjacent_items]), dim=-1)
            intermediate_gamma_feature = torch.sum(
                torch.mul(lgc_theta_user[users], self.get_multi_hop_vector(lgc_theta_item, intermediate_items)), dim=-1)
            distant_gamma_feature = torch.sum(
                torch.mul(lgc_theta_user[users], self.get_multi_hop_vector(lgc_theta_item, distant_items)), dim=-1)
            gamma_feature['adjacent'] = func.sigmoid(adjacent_gamma_feature)
            gamma_feature['intermediate'] = func.sigmoid(intermediate_gamma_feature)
            gamma_feature['distant'] = func.sigmoid(distant_gamma_feature)

            gamma_structure = {}
            adjacent_gamma_structure = []
            intermediate_gamma_structure = []
            distant_gamma_structure = []
            for index in range(self.flags_obj.n_layers):
                z1_user, z1_item = all_z[index]
                z2_user, z2_item = all_z[index + 1]

                adjacent_gamma_structure.append(torch.sum(torch.mul(z1_user[users], z2_item[adjacent_items]), dim=-1))
                adjacent_gamma_structure.append(torch.sum(torch.mul(z2_user[users], z1_item[adjacent_items]), dim=-1))
                intermediate_gamma_structure.append(
                    torch.sum(torch.mul(z1_user[users], self.get_multi_hop_vector(z2_item, intermediate_items)),
                              dim=-1))
                intermediate_gamma_structure.append(
                    torch.sum(torch.mul(z2_user[users], self.get_multi_hop_vector(z1_item, intermediate_items)),
                              dim=-1))
                distant_gamma_structure.append(
                    torch.sum(torch.mul(z1_user[users], self.get_multi_hop_vector(z2_item, distant_items)), dim=-1))
                distant_gamma_structure.append(
                    torch.sum(torch.mul(z2_user[users], self.get_multi_hop_vector(z1_item, distant_items)), dim=-1))

            adjacent_gamma_structure = torch.stack(adjacent_gamma_structure, dim=0)
            adjacent_gamma_structure = torch.mean(adjacent_gamma_structure, dim=0)
            intermediate_gamma_structure = torch.stack(intermediate_gamma_structure, dim=0)
            intermediate_gamma_structure = torch.mean(intermediate_gamma_structure, dim=0)
            distant_gamma_structure = torch.stack(distant_gamma_structure, dim=0)
            distant_gamma_structure = torch.mean(distant_gamma_structure, dim=0)
            gamma_structure['adjacent'] = adjacent_gamma_structure
            gamma_structure['intermediate'] = intermediate_gamma_structure
            gamma_structure['distant'] = distant_gamma_structure

            return gamma_feature, gamma_structure

        return gamma_genarate()

    def getEmbedding(self, users, positive_items, negative_items):
        users_embed_origin = self.user_embedding[users]
        positive_embed_origin = self.item_embedding[positive_items]
        negative_embed_origin = self.item_embedding[negative_items]

        return users_embed_origin, positive_embed_origin, negative_embed_origin

    def get_multi_hop_vector(self, embedding, items):
        zero_embedding = torch.zeros(embedding.shape[-1], device=self.flags_obj.device)
        multi_hop_embedding = embedding[items.long()]
        multi_hop_embedding[items == -1] = zero_embedding

        return multi_hop_embedding

    # def reparameter(self, mu, users, items):
    #     eps = torch.randn(size=mu.shape, device=self.flags_obj.device)
    #
    #     popularity_percent = self.popularity / self.num_users
    #     popularity_percent = popularity_percent[items.cpu().data.numpy()]
    #     popularity_alpha = popularity_percent / (2 - popularity_percent)
    #
    #     activity_percent = self.activity / self.num_items
    #     activity_percent = activity_percent[users.cpu().data.numpy()]
    #     activity_alpha = activity_percent / (2 - activity_percent)
    #
    #     alpha = popularity_alpha * activity_alpha
    #     alpha = torch.tensor(alpha, device=self.flags_obj.device)
    #     std = torch.pow(alpha, 0.25)
    #     # std = torch.sqrt(alpha)
    #
    #     z = mu + eps * std
    #
    #     return z

    # def information_loss(self, factual_rating):
    #     """
    #
    #     :param :
    #     :return: 获取CVIB信息正则化损失
    #     """
    #     users = torch.randint(low=0, high=self.num_users, size=(self.flags_obj.batch_size,),
    #                           device=self.flags_obj.device)
    #     items = torch.randint(low=0, high=self.num_items, size=(self.flags_obj.batch_size,),
    #                           device=self.flags_obj.device)
    #     users_embed = self.user_embedding[users.long()]
    #     items_embed = self.item_embedding[items.long()]
    #     counterfactual_scores = torch.mul(users_embed, items_embed)
    #     counterfactual_scores = torch.sum(counterfactual_scores, dim=-1)
    #     counterfactual_rating = func.sigmoid(counterfactual_scores)
    #
    #     factual_rating_average = factual_rating.mean()
    #     counterfactual_rating_average = counterfactual_rating.mean()
    #
    #     alpha = 1.0
    #     gamma = 0.1
    #
    #     a = - factual_rating_average * counterfactual_rating_average.log() - (1 - factual_rating_average) * (
    #             1 - counterfactual_rating_average).log()
    #     b = torch.mean(factual_rating * factual_rating.log())
    #     info_loss = alpha * (
    #             - factual_rating_average * counterfactual_rating_average.log() - (1 - factual_rating_average) * (
    #             1 - counterfactual_rating_average).log()) + gamma * torch.mean(
    #         factual_rating * factual_rating.log())
    #
    #     return info_loss

    def forward(self, users, adjacent_items, intermediate_items, distant_items):
        users_embed_origin = self.user_embedding[users.long()]
        adjacent_embed_origin = self.item_embedding[adjacent_items.long()]
        intermediate_embed_origin = self.get_multi_hop_vector(self.item_embedding, intermediate_items.long())
        distant_embed_origin = self.get_multi_hop_vector(self.item_embedding, distant_items.long())

        regular_loss1 = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                   adjacent_embed_origin.norm(2).pow(2) +
                                   intermediate_embed_origin.norm(2).pow(2) +
                                   distant_embed_origin.norm(2).pow(2)) / float(len(users))
        regular_loss2 = (1 / 2) * (
                (self.w1_item.norm(2).pow(2) + self.w2_item.norm(2).pow(2)) / float(self.num_items) +
                (self.w1_user.norm(2).pow(2) + self.w2_user.norm(2).pow(2)) / float(self.num_users)
        )
        regular_loss3 = (1 / 2) * (
                (self.theta_user.norm(2).pow(2)) / float(self.num_users) +
                (self.theta_item.norm(2).pow(2)) / float(self.num_items)
        )
        regular_loss = 10 * (self.flags_obj.weight_decay * (regular_loss1 + regular_loss3) + 0.1 * regular_loss2)

        # positive_scores = torch.sum(torch.mul(users_embed_origin, adjacent_embed_origin), dim=-1)
        # intermediate_scores = torch.sum(torch.mul(users_embed_origin, intermediate_embed_origin), dim=-1)
        # distant_scores = torch.sum(torch.mul(users_embed_origin, distant_embed_origin), dim=-1)

        # theta_, z1 = self.exposure_weight()
        # theta_user, theta_item, z_user, z_item = self.computer()
        gamma_feature, gamma_structure = self.computer(users, adjacent_items, intermediate_items, distant_items)
        intermediate_gamma = (gamma_structure['intermediate'] + gamma_feature['intermediate']) / 2
        distant_gamma = (gamma_structure['distant'] + gamma_feature['distant']) / 2

        gamma_feature_loss = torch.mean(torch.log(1 / gamma_feature['adjacent']))
        # gamma_feature_loss2 = torch.mean(- func.logsigmoid(gamma_feature['adjacent'] - gamma_feature['intermediate']))
        gamma_structure_loss = torch.mean(
            - func.logsigmoid(gamma_structure['intermediate'] - gamma_structure['distant']))

        # torch.mean(torch.log2(1 / gamma_structure['intermediate']))
        # torch.mean(-torch.log2(1 / gamma_structure['distant']))

        # gamma1_negative = torch.mul(theta_user[users], z_item[negative_items])
        # gamma1_negative = torch.sum(gamma1_negative, dim=-1)
        # gamma2_negative = torch.mul(z_user[users], theta_item[negative_items])
        # gamma2_negative = torch.sum(gamma2_negative, dim=-1)
        # gamma_negative = (func.sigmoid(gamma1_negative) + func.sigmoid(gamma2_negative)) / 2
        #
        # gamma1_positive = torch.mul(theta_user[users], z_item[positive_items])
        # gamma1_positive = torch.sum(gamma1_positive, dim=-1)
        # gamma2_positive = torch.mul(z_user[users], theta_item[positive_items])
        # gamma2_positive = torch.sum(gamma2_positive, dim=-1)
        # gamma_positive = (func.sigmoid(gamma1_positive) + func.sigmoid(gamma2_positive)) / 2

        positive_label = torch.ones(size=(adjacent_items.shape[0],), device=self.flags_obj.device)
        negative_label = torch.zeros(size=(adjacent_items.shape[0],), device=self.flags_obj.device)
        # gamma_positive = torch.ones(size=(adjacent_items.shape[0],), device=self.flags_obj.device)
        # gamma = torch.stack((gamma_positive, gamma_negative), dim=0)

        positive_embedding = adjacent_embed_origin
        intermediate_weight = intermediate_gamma / (intermediate_gamma + distant_gamma)
        intermediate_weight = intermediate_weight.unsqueeze(dim=-1)
        distant_weight = distant_gamma / (intermediate_gamma + distant_gamma)
        distant_weight = distant_weight.unsqueeze(dim=-1)
        negative_embedding = intermediate_weight * intermediate_embed_origin + distant_weight * distant_embed_origin

        positive_scores = torch.sum(torch.mul(users_embed_origin, positive_embedding), dim=-1)
        negative_scores = torch.sum(torch.mul(users_embed_origin, negative_embedding), dim=-1)
        positive_rating = func.sigmoid(positive_scores)
        negative_rating = func.sigmoid(negative_scores)

        rating = torch.stack((positive_rating, negative_rating), dim=0)
        label = torch.stack((positive_label, negative_label), dim=0)

        bce_loss = torch.mean(self.BCE(rating, label))

        # todo：exposure_loss加入消融实验，可以给负样本标签设置一个曝光先验值，比如1e-4?
        # exposure_loss = (self.BCE(gamma_positive, positive_label) + self.BCE(gamma_negative, negative_label)) / 2
        #
        # bpr_score = func.softplus(negative_scores - positive_scores)
        # weight_bpr_score = torch.mul(gamma_negative, bpr_score)
        #
        # loss = torch.mean(weight_bpr_score)
        # loss = loss + regular_loss + 0.1 * exposure_loss
        # todo: 增加曝光先验，点击样本曝光概率更高，未点击样本大概率未曝光，因此曝光先验概率值更小
        #  使用曝光先验的限制是否会造成过拟合的问题，点击样本曝光和未点击样本曝光的交叉熵只是为了纠正未点击的误差，并且在loss函数中只用到了未点击样本的曝光权重，所以应该不至于过拟合

        # exposure_loss = torch.mean(func.softplus(gamma_negative - gamma_positive))

        # todo:gamma增加重参数技巧减轻过拟合
        # items = torch.stack((positive_items, negative_items))
        # gamma = self.reparameter(gamma, users, items)

        # bce_loss = self.BCE(rating, label)
        # mf_loss = torch.mul(gamma, bce_loss)
        # mean_mf_loss = torch.mean(mf_loss)

        # mf_loss2 = func.binary_cross_entropy(rating, label, weight=gamma)
        # mean_mf_loss2 = torch.mean(mf_loss)

        # neg_rating = torch.stack((negative_rating, negative_rating))
        # b = self.BCE(neg_rating, label)
        # unknown_loss = torch.mul(1 - gamma, b)
        # mean_unknown_loss = torch.mean(unknown_loss)

        # info_loss = self.information_loss(rating)

        """ESCM2+BMSE
        ctr_loss = self.MSE(gamma.float(), label.float())
        ctr_loss = torch.mean(ctr_loss)
        ctcvr_loss = self.MSE(torch.mul(gamma, rating).float(), torch.mul(label, label).float())
        ctcvr_loss = torch.mean(ctcvr_loss)
        # ips_loss = torch.mul(torch.divide(label, gamma), bce_loss)
        ips_loss = torch.mul(torch.divide(label, gamma), self.MSE(rating.float(), label.float()))
        mean_ips_loss = torch.mean(ips_loss)
        bmse_loss = torch.divide(label, gamma) - torch.divide((1 - label), (1 - gamma))
        mean_bmse_loss = (torch.mean(torch.mul(bmse_loss, rating))) ** 2
        # mean_bmse_loss = (torch.mean(bmse_loss)) ** 2

        ips_v2_loss = mean_ips_loss + mean_bmse_loss + ctr_loss + ctcvr_loss
        """

        # bmse_loss = torch.divide(label, gamma) - torch.divide((1 - label), (1 - gamma))
        # mean_bmse_loss = (torch.mean(torch.mul(bmse_loss, rating))) ** 2
        # loss = mean_mf_loss + regular_loss + 0.1 * gamma_positive_kl - 0.01 * gamma_negative_penalty
        loss = bce_loss + gamma_feature_loss + 0.01 * gamma_structure_loss + regular_loss
        # loss = bce_loss + regular_loss

        # todo: 结合FAWMF，增加GCN如何达到自适应效果？
        return loss
        # return bce_loss, gamma_feature_loss, gamma_structure_loss, regular_loss


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
        regular_loss = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                  positive_embed_origin.norm(2).pow(2) +
                                  negative_embed_origin.norm(2).pow(2)) / float(len(users))

        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        # positive_rating = func.sigmoid(positive_scores)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)
        # negative_rating = func.sigmoid(negative_scores)

        # loss = torch.mean(func.softplus(negative_scores - positive_scores))
        loss = torch.mean(- func.logsigmoid(positive_scores - negative_scores))
        loss = loss + self.flags_obj.weight_decay * regular_loss

        return loss


class WMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(WMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))
        self.BCE = torch.nn.BCELoss()
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
        regular_loss = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                  positive_embed_origin.norm(2).pow(2) +
                                  negative_embed_origin.norm(2).pow(2)) / float(len(users))

        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        positive_rating = func.sigmoid(positive_scores)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)
        negative_rating = func.sigmoid(negative_scores)

        positive_label = torch.ones(size=(positive_rating.shape[0],), device=self.flags_obj.device)
        negative_label = torch.zeros(size=(negative_rating.shape[0],), device=self.flags_obj.device)

        loss = (self.BCE(positive_rating, positive_label) + self.BCE(negative_rating, negative_label)) / 2

        loss = loss + self.flags_obj.weight_decay * regular_loss
        # bpr_score = func.softplus(negative_scores - positive_scores)
        # loss = torch.mean(bpr_score)

        return loss


# EXMFgamma参数过多，num_users * num_items，内存不够
class EXMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(EXMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))
        self.gamma = Parameter(torch.randn([self.num_users, self.num_items]))

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
        regular_loss = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                  positive_embed_origin.norm(2).pow(2) +
                                  negative_embed_origin.norm(2).pow(2)) / float(len(users))

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
        loss = wmf_loss_mean + 0.1 * unknown_loss_mean + 0.01 * regular_loss

        return loss


class OneClassMF(nn.Module):
    def __init__(self, dataset: GraphDataset):
        super(OneClassMF, self).__init__()
        self.flags_obj = dataset.flags_obj

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.popularity = dataset.popularity
        self.trainInteractionSize = dataset.trainInteractionSize

        self.user_embedding = Parameter(torch.FloatTensor(self.num_users, self.flags_obj.embedding_dim))
        self.item_embedding = Parameter(torch.FloatTensor(self.num_items, self.flags_obj.embedding_dim))
        # self.BCE = torch.nn.BCELoss()
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
        regular_loss = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                  positive_embed_origin.norm(2).pow(2) +
                                  negative_embed_origin.norm(2).pow(2)) / float(len(users))

        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        # positive_rating = func.sigmoid(positive_scores)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)
        # negative_rating = func.sigmoid(negative_scores)

        # positive_label = torch.ones(size=(positive_rating.shape[0],), device=self.flags_obj.device)
        # negative_label = torch.zeros(size=(negative_rating.shape[0],), device=self.flags_obj.device)

        sparsity = self.trainInteractionSize / (self.num_users * self.num_items)

        popularity_percent = self.popularity / self.num_users
        alpha = popularity_percent[negative_items.cpu().data.numpy()]
        alpha = torch.tensor(alpha, device=self.flags_obj.device)
        # alpha = torch.pow(alpha, 0.25)

        # positive_mf_loss = torch.pow(1 - positive_rating, 2)
        # negative_mf_loss = torch.pow(1e-5 - negative_rating, 2)
        # negative_mf_loss = torch.mul(alpha / popularity_percent, negative_mf_loss)

        bpr_loss = func.softplus(negative_scores - positive_scores)
        weight_bpr_loss = torch.mul(alpha / sparsity, bpr_loss)
        weight_bpr_loss = torch.mean(weight_bpr_loss)
        loss = weight_bpr_loss + self.flags_obj.weight_decay * regular_loss
        # bpr_score = func.softplus(negative_scores - positive_scores)
        # loss = torch.mean(bpr_score)

        return loss
