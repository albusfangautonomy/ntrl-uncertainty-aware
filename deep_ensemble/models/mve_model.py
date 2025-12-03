import torch
import torch.nn as nn

class MVEModel(nn.Module):
    """
    Mean-Variance Estimation network
    outputs (mu, var)
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        var = torch.exp(log_var) + 1e-6   # numerical stability
        return mu, var
