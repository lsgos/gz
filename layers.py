from torch import nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ELU()
        )
    def forward(self, x):
        return x + self.body(x)

class LinearBlock(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.ELU(),
            nn.Linear(in_size, in_size),
            nn.ELU(),
            nn.Linear(in_size, in_size),
            nn.ELU()
        )
    def forward(self, x):
        return x + self.body(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
