import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder
from models.memory import MemoryModule


class MemoryAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.memory = MemoryModule(
            memory_size=50,
            feature_dim=512
        )

        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z_mem = self.memory(z)
        out = self.decoder(z_mem)
        return out