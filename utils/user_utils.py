import os

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from torch.utils import data


class TrainUserDataset(data.Dataset):
    """ Train User Data Loader: load training user-item interactions """

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_dir = os.path.join('data/', dataset)
        self.train_data_ui = self._load_train_data()
        self.user_list = list(range(self.n_users))

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        """ load user_id, binary vector over items """
        user = self.user_list[index]
        user_items = torch.from_numpy(self.train_data_ui[user, :].toarray()).squeeze()  # [I]
        return torch.from_numpy(np.array([user], dtype=np.int32)), user_items

    def _load_train_data(self):
        """ load training user-item interactions as a sparse matrix """
        path_ui = os.path.join(self.data_dir, 'train_ui.csv')
        df_ui = pd.read_csv(path_ui)
        self.n_users, self.n_items = df_ui['user'].max() + 1, df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']
        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(self.n_users, self.n_items))  # [# train users, I] sparse matrix

        print("# train users", self.n_users, "# items", self.n_items)
        return data_ui


class EvalUserDataset(data.Dataset):
    """ Eval User Data Loader: load val/test user-item interactions with fold-in and held-out item sets """

    def __init__(self, dataset, n_items, datatype='val'):
        self.dataset = dataset
        self.n_items = n_items
        self.data_dir = os.path.join('data/', dataset)
        self.data_tr, self.data_te = self._load_tr_te_data(datatype)

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        """ load user_id, fold-in items, held-out items """
        user = self.user_list[index]
        fold_in, held_out = self.data_tr[user, :].toarray(), self.data_te[user, :].toarray()  # [I], [I]
        return user, torch.from_numpy(fold_in).squeeze(), held_out.squeeze()  # user, fold-in items, fold-out items.

    def _load_tr_te_data(self, datatype='val'):
        """ load user-item interactions of val/test user sets as two sparse matrices of fold-in and held-out items """
        ui_tr_path = os.path.join(self.data_dir, '{}_ui_tr.csv'.format(datatype))
        ui_te_path = os.path.join(self.data_dir, '{}_ui_te.csv'.format(datatype))

        ui_df_tr, ui_df_te = pd.read_csv(ui_tr_path), pd.read_csv(ui_te_path)

        start_idx = min(ui_df_tr['user'].min(), ui_df_te['user'].min())
        end_idx = max(ui_df_tr['user'].max(), ui_df_te['user'].max())

        rows_tr, cols_tr = ui_df_tr['user'] - start_idx, ui_df_tr['item']
        rows_te, cols_te = ui_df_te['user'] - start_idx, ui_df_te['item']
        self.user_list = list(range(0, end_idx - start_idx + 1))

        ui_data_tr = sp.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.n_items))  # [# eval users, I] sparse matrix
        ui_data_te = sp.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.n_items))  # [# eval users, I] sparse matrix
        return ui_data_tr, ui_data_te
