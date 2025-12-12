import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkspaceEncoder2D(nn.Module):
    """
    2D version of the NTFields workspace encoder:
    - Input:  W0 (B, 1, H, W)   occupancy grid
    - Output: W  (B, K+1, H, W) multi-scale workspace feature volume
            where K is the number of CNN feature channels.
    """
    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        ch_in = in_channels
        ch_out = feature_channels

        for i in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,  # "same" spatial size
                )
            )
            layers.append(nn.ReLU(inplace=True))
            # Optional: you could add normalization here if needed
            ch_in = ch_out

        self.cnn = nn.Sequential(*layers)
        self.feature_channels = feature_channels

    def forward(self, W0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            W0: (B, 1, H, W) occupancy grid

        Returns:
            W:  (B, K+1, H, W), concatenation of W0 and CNN feature maps
        """
        # Sanity check
        if W0.dim() != 4:
            raise ValueError(f"Expected W0 to have shape (B, 1, H, W), got {W0.shape}")

        # Compute CNN features
        W1 = self.cnn(W0)  # (B, K, H, W)

        # Concatenate original occupancy channel to keep multi-scale info
        # W0: (B, 1, H, W), W1: (B, K, H, W) -> W: (B, K+1, H, W)
        W = torch.cat([W0, W1], dim=1)
        return W
