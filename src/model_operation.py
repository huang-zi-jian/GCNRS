import torch
from baselines_v3_3 import LightGCN, EXMF, FAWMF, BPR, WMF, LightGCL
from MIA_SP_v8 import MIA
from DRO_GCL import GSCL
import os
import torch.nn.functional as func
from dataloader import GraphDataset
import scipy.sparse as sp
import numpy as np


class ModelOperator(object):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(ModelOperator, self).__init__()
        self.flags_obj = flags_obj
        self.workspace = workspace
        self.dataset = dataset

        self.model = None
        self.device = torch.device(flags_obj.device)
        self.init_model()

    def init_model(self):
        raise NotImplementedError

    def save_model(self, epoch):
        ckpt_path = os.path.join(self.workspace, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')

        torch.save(self.model.state_dict(), model_path)

    def load_model(self, epoch):
        ckpt_path = os.path.join(self.workspace, 'ckpt')
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')

        # gpu加载还是cpu加载模型
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def getUsersRating(self, users):
        users = users.to(self.device)

        # all_users, all_items = self.model.computer()
        users_embed = self.model.user_embedding[users.long()]
        items_embed = self.model.item_embedding
        rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))
        return rating

    def get_loss(self, sample):
        users, positive_items, negative_items = sample

        users = users.to(self.device)
        positive_items = positive_items.to(self.device)
        negative_items = negative_items.to(self.device)

        loss = self.model(users, positive_items, negative_items)

        return loss


class lightGCN_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(lightGCN_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = LightGCN(self.dataset)
        self.model.to(self.device)

    def getUsersRating(self, users):
        users = users.to(self.device)

        all_users, all_items = self.model.computer()
        users_embed = all_users[users.long()]
        items_embed = all_items
        rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))
        return rating

    # def get_loss(self, sample):
    #     user, user_adjacent_item, items_pool, items_weight = sample
    #
    #     weak_items, strong_items = torch.split(items_pool, [1, 1], dim=1)
    #     weak_items = weak_items.squeeze()
    #     strong_items = strong_items.squeeze()
    #
    #     users = user.to(self.device)
    #     adjacent_items = user_adjacent_item.to(self.device)
    #     strong_items = strong_items.to(self.device)
    #
    #     loss = self.model(users, adjacent_items, strong_items)
    #
    #     return loss


class lightGCL_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(lightGCL_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = LightGCL(self.dataset)
        self.model.to(self.device)

    def getUsersRating(self, users):
        users = users.to(self.device)

        all_users, all_items, _, _ = self.model.computer()
        users_embed = all_users[users.long()]
        items_embed = all_items
        rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))

        return rating

    # def get_loss(self, sample):
    #     user, user_adjacent_item, items_pool, items_weight = sample
    #
    #     weak_items, strong_items = torch.split(items_pool, [1, 1], dim=1)
    #     weak_items = weak_items.squeeze()
    #     strong_items = strong_items.squeeze()
    #
    #     users = user.to(self.device)
    #     adjacent_items = user_adjacent_item.to(self.device)
    #
    #     loss = self.model(users, adjacent_items, weak_items, strong_items)
    #
    #     return loss


class FAWMF_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(FAWMF_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = FAWMF(self.dataset)
        self.model.to(self.device)


class MIA_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(MIA_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = MIA(self.dataset)
        self.model.to(self.device)

    def getUsersRating(self, users):
        users = users.to(self.device)
        # if self.flags_obj.dataset_name in ['coat']:
        #     users_preference, items_preference = self.model.pGcn()
        #     click_rating = func.sigmoid(torch.matmul(users_preference[users.long()], items_preference.t()))
        # else:
        users_preference, items_preference = self.model.pGcn()
        users_structure, items_structure = self.model.structure()

        users_exposure_embed = users_structure
        items_exposure_embed = items_structure

        users_embed = torch.cat((users_exposure_embed, users_preference), dim=-1)
        items_embed = torch.cat((items_exposure_embed, items_preference), dim=-1)
        click_rating = func.sigmoid(torch.matmul(users_embed[users.long()], items_embed.t()))

        return click_rating

    def get_loss(self, sample):
        user, user_adjacent_item, items_pool, items_weight = sample

        users = user.to(self.device)
        adjacent_items = user_adjacent_item.to(self.device)
        items_pool = items_pool.to(self.device)
        items_weight = items_weight.to(self.device)

        loss = self.model(users, adjacent_items, items_pool, items_weight)

        return loss


class GSCL_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(GSCL_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = GSCL(self.dataset)
        self.model.to(self.device)

    def getUsersRating(self, users):
        users = users.to(self.device)

        all_users, all_items, _, _ = self.model.computer()
        users_embed = all_users[users.long()]
        items_embed = all_items
        rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))

        return rating


# class AED_ModelOperator(ModelOperator):
#     def __init__(self, flags_obj, workspace, dataset: GraphDataset):
#         super(AED_ModelOperator, self).__init__(flags_obj, workspace, dataset)
#
#     def init_model(self):
#         self.model = AED(self.dataset)
#         self.model.to(self.device)
#
#     def getUsersRating(self, users):
#         users = users.to(self.device)
#
#         users_preference, items_preference = self.model.pGcn()
#
#         users_embed = users_preference
#         items_embed = items_preference
#         click_rating = func.sigmoid(torch.matmul(users_embed[users.long()], items_embed.t()))
#
#         return click_rating
#
#     def get_loss(self, sample):
#         user, user_adjacent_item, items_pool, items_weight = sample
#
#         users = user.to(self.device)
#         adjacent_items = user_adjacent_item.to(self.device)
#         items_pool = items_pool.to(self.device)
#         items_weight = items_weight.to(self.device)
#
#         loss = self.model(users, adjacent_items, items_pool, items_weight)
#
#         return loss


class EXMF_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(EXMF_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = EXMF(self.dataset)
        self.model.to(self.device)


class BPR_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(BPR_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = BPR(self.dataset)
        self.model.to(self.device)


class WMF_ModelOperator(ModelOperator):
    def __init__(self, flags_obj, workspace, dataset: GraphDataset):
        super(WMF_ModelOperator, self).__init__(flags_obj, workspace, dataset)

    def init_model(self):
        self.model = WMF(self.dataset)
        self.model.to(self.device)