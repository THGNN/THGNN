from dgl.generators import rand_bipartite
import numpy as np
import random


import torch
from torch import random
import torch.nn as nn


def set_seed(seed):
    np.random.seed(seed)
    # random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def user_id_to_idx(inters: np.array):
    inter_list = []
    userToidx = {}
    user_num = 0
    for i in range(len(inters)):
        if inters[i][0] in userToidx:
            userToidx[inters[i][0]] = userToidx[inters[i][0]]
        else:
            userToidx[inters[i][0]] = user_num
            user_num += 1
        if inters[i][1] in userToidx:
            userToidx[inters[i][1]] = userToidx[inters[i][1]]
        else:
            userToidx[inters[i][1]] = user_num
            user_num += 1
        inter_list.append(
            torch.tensor([userToidx[inters[i][0]], userToidx[inters[i][1]]]))
    return torch.stack(inter_list), userToidx

def rate_id_to_idx(write_inters: np.array, user_user_dict: dict):
    # write_inters: userID, objectID
    write_list = []
    writeToidx, objectToidx = {}, {}
    write_user_num = np.max(list(user_user_dict.values())) + 1
    write_object_num = 0
    for i in range(len(write_inters)):
        if write_inters[i][0] in user_user_dict:
            writeToidx[write_inters[i][0]] = user_user_dict[write_inters[i][0]]
        else:
            writeToidx[write_inters[i][0]] = write_user_num
            user_user_dict[write_inters[i][0]] = write_user_num
            write_user_num += 1
        if write_inters[i][1] in objectToidx:
            objectToidx[write_inters[i][1]] = objectToidx[write_inters[i][1]]
        else:
            objectToidx[write_inters[i][1]] = write_object_num
            write_object_num += 1
        write_list.append(
            torch.tensor([writeToidx[write_inters[i][0]],objectToidx[write_inters[i][1]]]))

    return torch.stack(write_list), user_user_dict, objectToidx

# def rate_id_to_idx(rate_inters: np.array, user_user_dict: dict, write_object_dict: dict):
#     # # rate_inters: userID, objectID
#     rate_list = []
#     rate_userToidx, rate_objectToidx = {}, {}
#     rate_user_num = np.max(list(user_user_dict.values())) + 1
#     rate_object_num = np.max(list(write_object_dict.values())) + 1
#     for i in range(len(rate_inters)):
#         if rate_inters[i][0] in user_user_dict:
#             rate_userToidx[rate_inters[i][0]] = user_user_dict[rate_inters[i][0]]
#         else:
#             rate_userToidx[rate_inters[i][0]] = rate_user_num
#             user_user_dict[rate_inters[i][0]] = rate_user_num
#             rate_user_num += 1
#         if rate_inters[i][1] in write_object_dict:
#             rate_objectToidx[rate_inters[i][1]] = write_object_dict[rate_inters[i][1]]
#         else:
#             rate_objectToidx[rate_inters[i][1]] = rate_object_num
#             write_object_dict[rate_inters[i][1]] = rate_object_num
#             rate_object_num += 1
#         rate_list.append(torch.tensor(
#             [rate_userToidx[rate_inters[i][0]], rate_objectToidx[rate_inters[i][1]]]))
#     return torch.stack(rate_list), user_user_dict, write_object_dict


class Decay(nn.Module):
    def __init__(self, cfg, w=2.0, method='log_decay'):
        self.cfg = cfg
        super().__init__()
        self.w = w
        self.method = method

    def exp_decay(self, delta_t):
        return torch.exp(- self.w * delta_t)

    def log_decay(self, delta_t):
        return 1./torch.log(2.7183 + self.w * delta_t)

    def rev_decay(self, delta_t):
        return 1./(1. + self.w * delta_t)

    def forward(self, delta_t):
        return getattr(self, self.method)(delta_t)
