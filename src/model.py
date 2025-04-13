import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math


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

        # gamma_positive_prior = torch.ones(size=(gamma_positive.shape[0],), device=self.flags_obj.device)
        # gamma_negative_prior = torch.normal(mean=0.3, std=0.1, size=(gamma_negative.shape[0],),
        #                                     device=self.flags_obj.device)
        # gamma_negative_prior[gamma_negative_prior <= 0] = 1e-2  # 曝光概率先验正态随机数不能小于 0
        # gamma_negative_prior[gamma_negative_prior > 1] = 1  # 曝光概率先验正态随机数不能大于 1
        # gamma_prior = torch.stack((gamma_positive_prior, gamma_negative_prior), dim=0)
        # gamma_prior_loss = self.BCE(gamma_prior, gamma)
        # mean_gamma_prior_loss = torch.mean(gamma_prior_loss)

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

    def computer(self):
        theta_user = func.softmax(self.theta_user, dim=-1)
        theta_item = func.softmax(self.theta_item, dim=-1)
        all_theta = torch.cat([theta_user, theta_item])

        if self.flags_obj.dropout:
            if self.training:
                graph_droped = self.__dropout(self.flags_obj.static_prob)
            else:
                graph_droped = self.Graph
        else:
            graph_droped = self.Graph

        # 聚合邻居节点以及子环信息 todo: 没有自环信息？
        z = torch.sparse.mm(graph_droped, all_theta)

        z_user, z_item = torch.split(z, [self.num_users, self.num_items], dim=0)
        z_user = func.sigmoid(z_user * self.w1_user + self.w2_user)
        z_item = func.sigmoid(z_item * self.w1_item + self.w2_item)

        return theta_user, theta_item, z_user, z_item

    def getEmbedding(self, users, positive_items, negative_items):
        users_embed_origin = self.user_embedding[users]
        positive_embed_origin = self.item_embedding[positive_items]
        negative_embed_origin = self.item_embedding[negative_items]

        return users_embed_origin, positive_embed_origin, negative_embed_origin

    def reparameter(self, mu, users, items):
        eps = torch.randn(size=mu.shape, device=self.flags_obj.device)

        popularity_percent = self.popularity / self.num_users
        popularity_percent = popularity_percent[items.cpu().data.numpy()]
        popularity_alpha = popularity_percent / (2 - popularity_percent)

        activity_percent = self.activity / self.num_items
        activity_percent = activity_percent[users.cpu().data.numpy()]
        activity_alpha = activity_percent / (2 - activity_percent)

        alpha = popularity_alpha * activity_alpha
        alpha = torch.tensor(alpha, device=self.flags_obj.device)
        std = torch.pow(alpha, 0.25)
        # std = torch.sqrt(alpha)

        z = mu + eps * std

        return z

    def information_loss(self, factual_rating):
        """

        :param :
        :return: 获取CVIB信息正则化损失
        """
        users = torch.randint(low=0, high=self.num_users, size=(self.flags_obj.batch_size,),
                              device=self.flags_obj.device)
        items = torch.randint(low=0, high=self.num_items, size=(self.flags_obj.batch_size,),
                              device=self.flags_obj.device)
        users_embed = self.user_embedding[users.long()]
        items_embed = self.item_embedding[items.long()]
        counterfactual_scores = torch.mul(users_embed, items_embed)
        counterfactual_scores = torch.sum(counterfactual_scores, dim=-1)
        counterfactual_rating = func.sigmoid(counterfactual_scores)

        factual_rating_average = factual_rating.mean()
        counterfactual_rating_average = counterfactual_rating.mean()

        alpha = 1.0
        gamma = 0.1

        a = - factual_rating_average * counterfactual_rating_average.log() - (1 - factual_rating_average) * (
                1 - counterfactual_rating_average).log()
        b = torch.mean(factual_rating * factual_rating.log())
        info_loss = alpha * (
                - factual_rating_average * counterfactual_rating_average.log() - (1 - factual_rating_average) * (
                1 - counterfactual_rating_average).log()) + gamma * torch.mean(
            factual_rating * factual_rating.log())

        return info_loss

    def forward(self, users, positive_items, negative_items):
        users_embed_origin = self.user_embedding[users.long()]
        positive_embed_origin = self.item_embedding[positive_items.long()]
        negative_embed_origin = self.item_embedding[negative_items.long()]

        regular_loss1 = (1 / 2) * (users_embed_origin.norm(2).pow(2) +
                                   positive_embed_origin.norm(2).pow(2) +
                                   negative_embed_origin.norm(2).pow(2)) / float(len(users))
        regular_loss2 = (1 / 2) * (
                (self.w1_item.norm(2).pow(2) + self.w2_item.norm(2).pow(2)) / float(self.num_items) +
                (self.w1_user.norm(2).pow(2) + self.w2_user.norm(2).pow(2)) / float(self.num_users)
        )
        regular_loss3 = (1 / 2) * (
                (self.theta_user.norm(2).pow(2)) / float(self.num_users) +
                (self.theta_item.norm(2).pow(2)) / float(self.num_items)
        )
        regular_loss = self.flags_obj.weight_decay * (regular_loss1 + regular_loss3) + 0.1 * regular_loss2

        positive_scores = torch.mul(users_embed_origin, positive_embed_origin)
        positive_scores = torch.sum(positive_scores, dim=-1)
        negative_scores = torch.mul(users_embed_origin, negative_embed_origin)
        negative_scores = torch.sum(negative_scores, dim=-1)

        # theta_, z1 = self.exposure_weight()
        theta_user, theta_item, z_user, z_item = self.computer()

        gamma1_negative = torch.mul(theta_user[users], z_item[negative_items])
        gamma1_negative = torch.sum(gamma1_negative, dim=-1)
        gamma2_negative = torch.mul(z_user[users], theta_item[negative_items])
        gamma2_negative = torch.sum(gamma2_negative, dim=-1)
        gamma_negative = (gamma1_negative + gamma2_negative) / 2

        gamma1_positive = torch.mul(theta_user[users], z_item[positive_items])
        gamma1_positive = torch.sum(gamma1_positive, dim=-1)
        gamma2_positive = torch.mul(z_user[users], theta_item[positive_items])
        gamma2_positive = torch.sum(gamma2_positive, dim=-1)
        gamma_positive = (gamma1_positive + gamma2_positive) / 2

        positive_label = torch.ones(size=(gamma_positive.shape[0],), device=self.flags_obj.device)
        negative_label = torch.zeros(size=(gamma_negative.shape[0],), device=self.flags_obj.device)

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
        # FAWMF损失函数
        positive_rating = func.sigmoid(positive_scores)
        negative_rating = func.sigmoid(negative_scores)
        rating = torch.stack((positive_rating, negative_rating), dim=0)
        label = torch.stack((positive_label, negative_label), dim=0)
        gamma = torch.stack((gamma_positive, gamma_negative), dim=0)

        # todo:gamma增加重参数技巧减轻过拟合
        items = torch.stack((positive_items, negative_items))
        # gamma = self.reparameter(gamma, users, items)

        # mf_loss = torch.pow(rating - label, 2)
        # mf_loss = torch.mul(gamma, mf_loss)
        # mean_mf_loss = torch.mean(mf_loss)
        unknown_loss = torch.pow(1e-5 - label, 2)
        unknown_loss = torch.mul(1 - gamma, unknown_loss)
        mean_unknown_loss = torch.mean(unknown_loss)

        bce_loss = self.BCE(rating, label)
        mf_loss = torch.mul(gamma, bce_loss)
        mean_mf_loss = torch.mean(mf_loss)

        gamma_positive_kl = torch.mean(torch.log2(1 / gamma_positive))
        gamma_negative_penalty = torch.mean(gamma_negative * torch.log2(1 / gamma_negative))

        # mf_loss2 = func.binary_cross_entropy(rating, label, weight=gamma)
        # mean_mf_loss2 = torch.mean(mf_loss)

        # neg_rating = torch.stack((negative_rating, negative_rating))
        # b = self.BCE(neg_rating, label)
        # unknown_loss = torch.mul(1 - gamma, b)
        # mean_unknown_loss = torch.mean(unknown_loss)

        # loss = mean_mf_loss + 0.1 * mean_unknown_loss
        fawmf_loss = mean_mf_loss + 0.1 * mean_unknown_loss

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
        loss = mean_mf_loss + regular_loss + 0.01 * gamma_positive_kl - 0.01 * gamma_negative_penalty
        # loss = fawmf_loss

        # todo: 结合FAWMF，增加GCN如何达到自适应效果？
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
