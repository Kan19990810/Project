import torch

import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, inp):
        x = self.se(inp)
        return inp * x


class SPAModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SPAModule, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool1d(4)
        self.se = nn.Sequential(
            nn.Conv1d(channels*7, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, inp):
        b, c, _ = inp.size()
        x1 = self.avg_pool1(inp)
        x2 = self.avg_pool2(inp).view(b, c * 2, 1)
        x3 = self.avg_pool3(inp).view(b, c * 4, 1)
        x  = self.se(torch.cat((x1, x2, x3), dim=1))
        return inp * x


class ECAModule(nn.Module):
    def __init__(self, k_size=5):
        super(ECAModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=int(k_size//2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        # (batch, channels , 1)
        x = self.pool(inp)
        # (batch, 1, channels) -> (batch, channels, 1)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.sigmoid(x)
        return inp * x

