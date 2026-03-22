import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule(nn.Module):

    def __init__(self,
                 memory_size=50,
                 feature_dim=512):
        super().__init__()
        self.memory = nn.Parameter(
            torch.randn(
                memory_size,
                feature_dim
            )
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.view(
            B, C, -1
        )

        x_flat = x_flat.permute(
            0, 2, 1
        )

        memory_norm = F.normalize(
            self.memory,
            dim=1
        )

        x_norm = F.normalize(
            x_flat,
            dim=2
        )

        similarity = torch.matmul(
            x_norm,
            memory_norm.t()
        )

        weights = F.softmax(
            similarity,
            dim= -1
        )

        memory_output = torch.matmul(
            weights,
            self.memory
        )

        memory_output = memory_output.permute(0, 2, 1)
        memory_output = memory_output.view( B, C, H, W)
        return memory_output