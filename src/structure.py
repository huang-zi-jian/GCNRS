import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from dataloader import GraphDataset
from torch.nn.parameter import Parameter
import math
import pickle


def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=-1)[:, None], b.norm(dim=-1)[:, None]
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
        self.origin_Graph = dataset.origin_Graph
        self.embedding_dim = self.flags_obj.embedding_dim

        self.threshold = 0.1
        # self.num_community = self.flags_obj.num_community

        self.user_structure = Parameter(torch.FloatTensor(self.num_users, self.embedding_dim))
        self.item_structure = Parameter(torch.FloatTensor(self.num_items, self.embedding_dim))

        self.weight = Parameter(torch.FloatTensor(1, self.embedding_dim))

        self.distance = nn.PairwiseDistance(p=2)
        self.cos_similar = nn.CosineSimilarity()
        self.init_weight()

    def init_weight(self):
        # nn.init.normal_(self.user_embedding.data, std=0.1)
        # nn.init.normal_(self.item_embedding.data, std=0.1)
        stdv = 1. / math.sqrt(self.embedding_dim)
        self.user_structure.data.uniform_(-stdv, stdv)
        self.item_structure.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.weight)

    def similarity(self):
        # user_similarity_matrix = torch.zeros((self.num_users, self.num_users)).to(device=self.flags_obj.device)
        # similarity_matrix = torch.zeros((self.num_users, self.num_items)).to(device=self.flags_obj.device)
        # similarity_matrix = torch.zeros((self.num_users, self.num_items)).to(device=self.flags_obj.device)

        # zero_lines = torch.nonzero(torch.sum(self.user_structure, dim=-1) == 0)
        # if len(zero_lines) > 0:
        #     self.user_structure[zero_lines, :] += 1e-8

        weight_user_structure = self.user_structure * self.weight
        weight_item_structure = self.item_structure * self.weight

        user_similarity = cos_sim(weight_user_structure, weight_user_structure)
        item_similarity = cos_sim(weight_item_structure, weight_item_structure)
        user_item_similarity = cos_sim(weight_user_structure, weight_item_structure)

        user_similarity = torch.where(user_similarity < self.threshold, torch.zeros_like(user_similarity),
                                      user_similarity)
        item_similarity = torch.where(item_similarity < self.threshold, torch.zeros_like(item_similarity),
                                      item_similarity)
        user_item_similarity = torch.where(user_item_similarity < self.threshold,
                                           torch.zeros_like(user_item_similarity), user_item_similarity)

        similarity_based_user = user_similarity.mm(self.origin_Graph)
        similarity_based_item = self.origin_Graph.mm(item_similarity)

        similarity_matrix = similarity_based_user + similarity_based_item + user_item_similarity

        return similarity_matrix

    def forward(self):
        # users_map = self.user_structure
        # items_map = self.item_structure
        users_map = func.normalize(self.user_structure, p=2, dim=-1)
        items_map = func.normalize(self.item_structure, p=2, dim=-1)

        # return users_structure, adjacent_items_structure, weak_items_structure, strong_items_structure
        return users_map, items_map


if __name__ == '__main__':
    with open('../dataset/yelp-pkl/node_features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('../dataset/yelp-pkl/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open('../dataset/yelp-pkl/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('../dataset/yelp-pkl/meta_data.pkl', 'rb') as f:
        meta_data = pickle.load(f)
        print()

    features = torch.from_numpy(features).type(torch.FloatTensor)
    features = func.normalize(features, dim=1, p=1)
    print()
