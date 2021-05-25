import numpy as np
from numpy.core.arrayprint import dtype_short_repr
import pandas as pd

import torch.utils.data as data 

class writeDataset(data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        # user_write_object = np.loadtxt(cfg['mc_path'], dtype=str, usecols=(0, 1), max_rows=10)  # objectID, userID
        # user_rate_object_part1 = np.loadtxt(cfg['rating_path'], dtype=str, skiprows=0, usecols=(0, 1, 2, 4), max_rows=845)
        # user_rate_object_part2 = np.loadtxt(cfg['rating_path'], dtype=str, skiprows=845, usecols=(0, 1, 2, 5))  
        # user_rate_object = np.concatenate((user_rate_object_part1, user_rate_object_part2))  # objectID, userID, rating, date
        # user_rate_user = np.loadtxt(cfg['user_path'], dtype=str)  # MyID, otherID(being trusted/distrusted),  1(trust)-1(distrust), date
        self.user_write_object = pd.read_csv(cfg['mc_path'], skiprows=0, dtype=str)
        # self.user_rate_object = pd.read_csv(cfg['rating_path'], skiprows=0, dtype=str)
        # self.user_rate_user = pd.read_csv(cfg['user_path'], skiprows=0, dtype=str)
        # self.user_rate_object['date'] = pd.to_datetime(self.user_rate_object.date)
        # self.user_rate_user['date'] = pd.to_datetime(self.user_rate_user.date) 


    def __len__(self):
        return self.user_write_object.shape[0]

    def __getitem__(self, index:int):
        return self.user_write_object[index, :]
