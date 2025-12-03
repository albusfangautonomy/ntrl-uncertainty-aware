import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """
    Example dataset wrapper:
    X: (N, input_dim)
    y: (N, 1)
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
