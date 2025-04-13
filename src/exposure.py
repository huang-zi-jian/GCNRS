import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed


class WMF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super(WMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)
        self.user_embedding.weight.data *= 0.1
        self.item_embedding.weight.data *= 0.1

        self.affine_output = nn.Linear(in_features=self.dim, out_features=1)
        self.logistic = nn.Sigmoid()

    def init_weight(self):
        pass

    def allprediction(self):
        rating = self.logistic(self.user_embedding.weight.mm(self.item_embedding.weight.t()))

        return rating

    def getem(self):
        return self.user_embedding.weight.detach(), self.item_embedding.weight.detach()

    def itemprediction(self, samitem):
        with torch.no_grad():
            rating = self.logistic(self.user_embedding.weight.mm(self.item_embedding(samitem).t()))

            return rating

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        element_product = torch.mul(user_emb, item_emb)
        logits = torch.sum(element_product, dim=1)
        rating = self.logistic(logits)

        return rating