import os

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from torch.utils import data


class TrainGroupDataset(data.Dataset):
    """ Train Group Data Loader: load training group-item interactions and individual user item interactions """

    def __init__(self, dataset, n_items, negs_per_group):
        self.dataset = dataset
        self.n_items = n_items
        self.negs_per_group = negs_per_group
        self.data_dir = os.path.join('data/', dataset)
        self.user_data = self._load_user_data()
        self.group_data, self.group_users = self._load_group_data()
        self.group_inputs = [self.user_data[self.group_users[g]] for g in self.groups_list]

    def __len__(self):
        return len(self.groups_list)

    def get_corrupted_users(self, group):
        """ negative user sampling per group (eta balances item-biased and random sampling) """
        eta = 0.5
        p = np.ones(self.n_users + 1)
        p[self.group_users[group]] = 0
        p = normalize([p], norm='l1')[0]
        item_biased = normalize(self.user_data[:, self.group_data[group].indices].sum(1).squeeze(), norm='l1')[0]
        p = eta * item_biased + (1 - eta) * p
        negative_users = torch.multinomial(torch.from_numpy(p), self.negs_per_group)
        return negative_users

    def __getitem__(self, index):
        """ load group_id, padded group users, mask, group items, group member items, negative user items """
        group = self.groups_list[index]
        user_ids = torch.from_numpy(np.array(self.group_users[group], np.int32))  # [G] group member ids
        group_items = torch.from_numpy(self.group_data[group].toarray().squeeze())  # [I] items per group

        corrupted_group = self.get_corrupted_users(group)  # [# negs]
        corrupted_user_items = torch.from_numpy(self.user_data[corrupted_group].toarray().squeeze())  # [# negs, I]

        # group mask to create fixed-size padded groups.
        group_length = self.max_group_size - list(user_ids).count(self.padding_idx)
        group_mask = torch.from_numpy(np.concatenate([np.zeros(group_length, dtype=np.float32), (-1) * np.inf *
                                                      np.ones(self.max_group_size - group_length,
                                                              dtype=np.float32)]))  # [G]

        user_items = torch.from_numpy(self.group_inputs[group].toarray())  # [G, |I|] group member items

        return torch.tensor([group]), user_ids, group_mask, group_items, user_items, corrupted_user_items

    def _load_group_data(self):
        """ load training group-item interactions as a sparse matrix and user-group memberships """
        path_ug = os.path.join(self.data_dir, 'group_users.csv')
        path_gi = os.path.join(self.data_dir, 'train_gi.csv')

        df_gi = pd.read_csv(path_gi)  # load training group-item interactions.
        start_idx, end_idx = df_gi['group'].min(), df_gi['group'].max()
        self.n_groups = end_idx - start_idx + 1
        rows_gi, cols_gi = df_gi['group'] - start_idx, df_gi['item']

        data_gi = sp.csr_matrix((np.ones_like(rows_gi), (rows_gi, cols_gi)), dtype='float32',
                                shape=(self.n_groups, self.n_items))  # [# groups,  I] sparse matrix.

        df_ug = pd.read_csv(path_ug).astype(int)  # load user-group memberships.
        df_ug_train = df_ug[df_ug.group.isin(range(start_idx, end_idx + 1))]
        df_ug_train = df_ug_train.sort_values('group')  # sort in ascending order of group ids.
        self.max_group_size = df_ug_train.groupby('group').size().max()  # max group size denoted by G

        g_u_list_train = df_ug_train.groupby('group')['user'].apply(list).reset_index()
        g_u_list_train['user'] = list(map(lambda x: x + [self.padding_idx] * (self.max_group_size - len(x)),
                                          g_u_list_train.user))
        data_gu = np.squeeze(np.array(g_u_list_train[['user']].values.tolist()))  # [# groups, G] with padding.
        self.groups_list = list(range(0, end_idx - start_idx + 1))

        assert len(df_ug_train['group'].unique()) == self.n_groups
        print("# training groups: {}, # max train group size: {}".format(self.n_groups, self.max_group_size))

        return data_gi, data_gu

    def _load_user_data(self):
        """ load user-item interactions of all users that appear in training groups, as a sparse matrix """
        df_ui = pd.DataFrame()
        train_path_ui = os.path.join(self.data_dir, 'train_ui.csv')
        df_train_ui = pd.read_csv(train_path_ui)
        df_ui = df_ui.append(df_train_ui)

        # include users from the (fold-in item set) of validation and test sets of user-item data.
        val_path_ui = os.path.join(self.data_dir, 'val_ui_tr.csv')
        df_val_ui = pd.read_csv(val_path_ui)
        df_ui = df_ui.append(df_val_ui)

        test_path_ui = os.path.join(self.data_dir, 'test_ui_tr.csv')
        df_test_ui = pd.read_csv(test_path_ui)
        df_ui = df_ui.append(df_test_ui)

        self.n_users = df_ui['user'].max() + 1
        self.padding_idx = self.n_users  # padding idx for user when creating groups of fixed size.
        assert self.n_items == df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']

        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(self.n_users + 1, self.n_items))  # [U, I] sparse matrix
        return data_ui


class EvalGroupDataset(data.Dataset):
    """ Eval Group Data Loader: load val/test group-item interactions and individual user item interactions  """

    def __init__(self, dataset, n_items, padding_idx, datatype='val'):
        self.dataset = dataset
        self.padding_idx = padding_idx
        self.n_items = n_items
        self.data_dir = os.path.join('data/', dataset)
        self.eval_groups_list = []
        self.user_data = self._load_user_data(datatype)
        self.eval_group_data, self.eval_group_users = self._load_group_data(datatype)

    def __len__(self):
        return len(self.eval_groups_list)

    def __getitem__(self, index):
        """ load group_id, padded group users, mask, group items, group member items """
        group = self.eval_groups_list[index]
        user_ids = self.eval_group_users[group]  # [G]
        length = self.max_gsize - list(user_ids).count(self.padding_idx)
        mask = torch.from_numpy(np.concatenate([np.zeros(length, dtype=np.float32), (-1) * np.inf *
                                                np.ones(self.max_gsize - length, dtype=np.float32)]))  # [G]
        group_items = torch.from_numpy(self.eval_group_data[group].toarray().squeeze())  # [I]
        user_items = torch.from_numpy(self.user_data[user_ids].toarray().squeeze())  # [G, I]

        return torch.tensor([group]), torch.tensor(user_ids), mask, group_items, user_items

    def _load_user_data(self, datatype):
        """ load all user-item interactions of users that occur in val/test groups, as a sparse matrix """
        df_ui = pd.DataFrame()
        train_path_ui = os.path.join(self.data_dir, 'train_ui.csv')
        df_train_ui = pd.read_csv(train_path_ui)
        df_ui = df_ui.append(df_train_ui)

        val_path_ui = os.path.join(self.data_dir, 'val_ui_tr.csv')
        df_val_ui = pd.read_csv(val_path_ui)
        df_ui = df_ui.append(df_val_ui)

        if datatype == 'val' or datatype == 'test':
            # include eval user set (tr) items (since they might occur in evaluation set)
            test_path_ui = os.path.join(self.data_dir, 'test_ui_tr.csv')
            df_test_ui = pd.read_csv(test_path_ui)
            df_ui = df_ui.append(df_test_ui)

        n_users = df_ui['user'].max() + 1
        assert self.n_items == df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']
        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(n_users + 1, self.n_items))  # [# users, I] sparse matrix
        return data_ui

    def _load_group_data(self, datatype):
        """ load val/test group-item interactions as a sparse matrix and user-group memberships """
        path_ug = os.path.join(self.data_dir, 'group_users.csv')
        path_gi = os.path.join(self.data_dir, '{}_gi.csv'.format(datatype))

        df_gi = pd.read_csv(path_gi)  # load group-item interactions
        start_idx, end_idx = df_gi['group'].min(), df_gi['group'].max()
        self.n_groups = end_idx - start_idx + 1
        rows_gi, cols_gi = df_gi['group'] - start_idx, df_gi['item']
        data_gi = sp.csr_matrix((np.ones_like(rows_gi), (rows_gi, cols_gi)), dtype='float32',
                                shape=(self.n_groups, self.n_items))  # [# eval groups, I] sparse matrix

        df_ug = pd.read_csv(path_ug)  # load user-group memberships
        df_ug_eval = df_ug[df_ug.group.isin(range(start_idx, end_idx + 1))]
        df_ug_eval = df_ug_eval.sort_values('group')  # sort in ascending order of group ids
        self.max_gsize = df_ug_eval.groupby('group').size().max()  # max group size denoted by G
        g_u_list_eval = df_ug_eval.groupby('group')['user'].apply(list).reset_index()
        g_u_list_eval['user'] = list(map(lambda x: x + [self.padding_idx] * (self.max_gsize - len(x)),
                                         g_u_list_eval.user))
        data_gu = np.squeeze(np.array(g_u_list_eval[['user']].values.tolist(), dtype=np.int32))  # [# groups, G]
        self.eval_groups_list = list(range(0, end_idx - start_idx + 1))
        return data_gi, data_gu
