import torch.nn as nn

class ContextNet(nn.Module):
    def __init__(self, u_dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * h_dim)
        )

    def forward(self, u):
        gamma, beta = self.net(u).chunk(2, dim=-1)
        return gamma, beta

class FiLM(nn.Module):
    def forward(self, h, gamma, beta):
        return gamma * h + beta
