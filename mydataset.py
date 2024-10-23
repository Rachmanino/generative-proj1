import torch
import json
from torch.utils.data import Dataset, DataLoader
from tokenizer import tokenizer

class MyTrainDataset(Dataset):
    """Construct a trainning dataset from data/train.json"""

    def __init__(self):
        super(MyTrainDataset, self).__init__()
        self.path = 'data/train.json'
        with open(self.path) as f:
            self._data = json.load(f) # [{'text': '...',}, ...]
            self.data = [d['text'] for d in self._data] # ['...', ...]


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
