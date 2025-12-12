import torch
import torch.nn as nn

class CSpaceEncoder(nn.Module):
    """
    f(q): MLP encoder for robot configuration.
    """
    def __init__(self, q_dim=2, hidden_dim=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, q):
        return self.net(q)  # (B, out_dim)
