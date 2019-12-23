import pandas as pd
import torch
from torch.utils.data import Dataset


class MachineDataset(Dataset):

    """ Custom dataset class to load data """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep=',', decimal='.')
        self.data = torch.from_numpy(self.df[self.df.columns[:-1]].values).float()
        self.targets = torch.from_numpy(self.df[self.df.columns[-1]].values).long()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx, :-1].astype('float')
        y = self.df.iloc[idx, -1].astype('int')
        return torch.FloatTensor(x), torch.tensor(y, dtype=torch.long)
