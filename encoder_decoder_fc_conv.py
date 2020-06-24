import torch.nn as nn
import torch
from layers import ConvBlock, Flatten, LinearBlock
class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    """
    def __init__(self,
                 insize=56, z_dim=10):
        super().__init__()
        self.insize = insize
        self.linear_size = int((insize/8)**2)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool2d(2),
            ConvBlock(32),
            nn.Conv2d(32,16,kernel_size=3, padding=1),
            nn.ELU(),
            ConvBlock(16),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.AvgPool2d(2),
            nn.ELU()
        )
        self.loc = nn.Linear(self.linear_size, z_dim)
        self.scale = nn.Linear(self.linear_size, z_dim)
        
    def forward(self, x):
        x = x - 0.222
        x = x / 0.156
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        z_loc = self.loc(x)
        z_scale = torch.exp(self.scale(x))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.outsize = outsize
        self.flatten_size = int((self.outsize)**2)
        self.net = nn.Sequential(
            nn.Linear(z_dim, int(self.flatten_size/8)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/8), int(self.flatten_size/4)),
            LinearBlock(int(self.flatten_size/4)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/4), int(self.flatten_size/2)),
            LinearBlock(int(self.flatten_size/2)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/2), self.flatten_size),
            nn.Sigmoid(),
        )
        
    def forward(self, z):
        z = self.net(z)
        z = z.reshape(z.shape[0], self.outsize, self.outsize)
        return z
