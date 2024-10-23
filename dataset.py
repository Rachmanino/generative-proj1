import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """Construct a custom dataset from data/*.json"""

    def __init__(self):
        super(MyDataset, self).__init__()
        raise NotImplementedError   

    def __getitem__(self, index):
        return 

    def __len__(self):
        return 
