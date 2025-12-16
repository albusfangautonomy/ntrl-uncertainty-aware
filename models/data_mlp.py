import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable


class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        data: Tensor of shape (N, D)
        """
        self.data = Variable(data)

    def send_device(self, device):
        self.data = self.data.to(device)

    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return self.data.shape[0]


def Database(PATH):
    """
    Expected files in PATH:
      - sampled_points.npy      (N, 2*dim)
      - speed_mean.npy          (N, 2)
      - speed_var.npy           (N, 2)
      - normal_mean.npy         (N, 2*dim)
      - normal_var.npy          (N, 2*dim)
    """

    # Load data
    points = np.load(f'{PATH}/sampled_points.npy')
    speed_mean = np.load(f'{PATH}/speed_mean.npy')
    speed_var  = np.load(f'{PATH}/speed_var.npy')
    normal_mean = np.load(f'{PATH}/normal_mean.npy')
    normal_var  = np.load(f'{PATH}/normal_var.npy')

    # Basic sanity checks (very helpful during transition)
    assert points.shape[0] == speed_mean.shape[0] == speed_var.shape[0], \
        "Mismatch in number of samples (points / speed)"
    assert normal_mean.shape == normal_var.shape, \
        "Normal mean/var shape mismatch"

    print("Loaded dataset:")
    print(" points       :", points.shape)
    print(" speed_mean  :", speed_mean.shape)
    print(" speed_var   :", speed_var.shape)
    print(" normal_mean :", normal_mean.shape)
    print(" normal_var  :", normal_var.shape)

    # Convert to torch tensors
    points = Tensor(points)
    speed_mean = Tensor(speed_mean)
    speed_var  = Tensor(speed_var)
    normal_mean = Tensor(normal_mean)
    normal_var  = Tensor(normal_var)

    # Concatenate into one tensor
    data = torch.cat(
        (points,
         speed_mean,
         speed_var,
         normal_mean,
         normal_var),
        dim=1
    )

    return _numpy2dataset(data)
