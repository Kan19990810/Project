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


class CBAModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(CBAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.ca = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            )
        self.sa = nn.Conv1d(2,1,kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # channel attention
        x1 = self.avg_pool(input)
        x1 = self.ca(x1)
        x2 = self.max_pool(input)
        x2 = self.ca(x2)
        x = x1 + x2
        x = self.sigmoid(x)
        x_ca = input * x

        # spatial attention
        y1 = self.avg_pool(x_ca.transpose(-1, -2)).transpose(-1, -2)
        y2 = self.max_pool(x_ca.transpose(-1, -2)).transpose(-1, -2)
        y = torch.cat((y1, y2), dim=1)
        y = self.sa(y)
        y = self.sigmoid(y)
        x_sa = x_ca * y
        return x_sa


class frequencyBandAtten(nn.Module):
    def __init__(self, channels, r=4):
        super(frequencyBandAtten, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels//r, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels//r, channels, kernel_size=1, padding=0),
        )
        self.spaModule = nn.Conv1d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        # channel attention
        x1 = self.mlp(self.avg_pool(inp))
        x2 = self.mlp(self.max_pool(inp))
        channel_out = self.sigmoid(x1 + x2)
        x = channel_out * inp

        # spatial attention
        # y1 = self.avg_pool(x.transpose(-1, -2)).transpose(-1, -2)
        # y2 = self.max_pool(x.transpose(-1, -2)).transpose(-1, -2)
        y1, _ = torch.max(x, dim=1, keepdim=True)
        y2 = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([y1, y2], dim=1)
        spatial_out = self.sigmoid(self.spaModule(y))
        x = spatial_out * x
        return x
