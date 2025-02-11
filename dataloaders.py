import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import scanpy as sc


class BulkRNADataset(Dataset):
    def __init__(self, path, split, seed=None):
        self.split = split
        df = pd.read_csv(path)
        cellline_R = df[df['CISPLATIN'] == 'resistant']
        cellline_S = df[df['CISPLATIN'] == 'sensitive']

        R_train, R_test = train_test_split(cellline_R, train_size=-.8, random_state=seed)
        S_train, S_test = train_test_split(cellline_S, train_size=-.8, random_state=seed)
        R_train, R_val = train_test_split(R_train, test_size=0.25, random_state=seed)
        S_train, S_val = train_test_split(S_train, test_size=0.25, random_state=seed)

        df_train = pd.concat([R_train, S_train], axis=0)
        df_val = pd.concat([R_val, S_val], axis=0)
        df_test = pd.concat([R_test, S_test], axis=0)

        if split == 'train':
            self.df = df_train
        elif split == 'val':
            self.df = df_val
        elif split == 'test':
            self.df = df_test
        else:
            raise ValueError()

        self.df = self.df.sample(frac=1, random_state=seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        X, y = row.drop(columns='CISPLATIN'), row['CISPLATIN']
        return X, y


class SingleCellTumorDataset(Dataset):
    def __init__(self, path, split, seed=None):
        super().__init__()
        adata = sc.read_h5ad(path)
        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        X_train, X_test = train_test_split(df, train_size=0.8, random_state=seed)
        X_train, X_val = train_test_split(X_train, train_size=0.25, random_state=seed)
        if split == 'train':
            self.df = X_train
        elif split == 'val':
            self.df = X_val
        elif split == 'test':
            self.df = X_test
        else:
            raise ValueError()
        self.df = self.df.sample(frac=1, random_state=seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return self.df.iloc[index, :]


class SingleCellLineData(Dataset):
    def __init__(self, path, split, seed=None):
        super().__init__()
        adata = sc.read_h5ad(path)
        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df['label'] = adata[:, adata.obs['sensitive']]

        cell_lines_R = df[df['label'] == 0]
        cell_lines_S = df[df['label'] == 1]

        R_train, R_test = train_test_split(cell_lines_R, train_size=-.8, random_state=seed)
        S_train, S_test = train_test_split(cell_lines_S, train_size=-.8, random_state=seed)
        R_train, R_val = train_test_split(R_train, test_size=0.25, random_state=seed)
        S_train, S_val = train_test_split(S_train, test_size=0.25, random_state=seed)

        df_train = pd.concat([R_train, S_train], axis=0)
        df_val = pd.concat([R_val, S_val], axis=0)
        df_test = pd.concat([R_test, S_test], axis=0)

        if split == 'train':
            self.df = df_train
        elif split == 'val':
            self.df = df_val
        elif split == 'test':
            self.df = df_test
        else:
            raise ValueError()

        self.df = self.df.sample(frac=1, random_state=seed)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return self.df.iloc[index, :]

