import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from models import AttentionBlocks


class PreEmphasis(nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            # flipped_filter 维度 [1, 1, 2]
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inp: torch.tensor) -> torch.tensor:
        # inp 维度 [length, 1]
        inp = inp.unsqueeze(1)
        inp = F.pad(inp, (1, 0), 'reflect')
        return F.conv1d(inp, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class dkconv(nn.Module):
    def __init__(self, channels, kernel_size, M=2, r=16, G=32, L=32):
        super(dkconv, self).__init__()
        d = max(int(channels / r), L)
        self.M = M
        self.channels = channels
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels * 2, d, kernel_size=1, stride=1),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=False))
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1,
                          dilation=i + 1, padding=math.floor(kernel_size / 2) * (i + 1),
                          groups=G, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU()))
            self.fcs.append(nn.Conv1d(d, channels, kernel_size=1, stride=1))

    def forward(self, x):
        batch_size = x.shape[0]
        # M * (batch_size, channels, time)
        feats = [conv(x) for conv in self.convs]
        # (batch_size,  2 * channels, time)
        feats = torch.cat(feats, dim=1)
        # (batch_size, M=2, channels, time)
        feats = feats.view(batch_size, self.M, self.channels, feats.shape[2])
        # (batch, channels, time)
        feats_U = torch.sum(feats, dim=1)
        # (batch, channels, 1)
        fea_mu = feats_U.mean(dim=2).squeeze(-1)
        fea_sg = feats_U.std(dim=2).squeeze(-1)
        # (batch, channels * 2)
        feats_S = torch.cat((fea_mu, fea_sg), dim=1).unsqueeze(-1)
        # (batch, d = channels / r)
        feats_Z = self.fc(feats_S)
        # M * (batch, channels)
        atten = [fc(feats_Z) for fc in self.fcs]
        # (batch, channels * 2)
        atten = torch.cat(atten, dim=1)
        # (batch, M = 2, channels, 1)
        atten = atten.view(batch_size, self.M, self.channels, 1)
        atten = self.softmax(atten)
        # (batch, channels, time)
        feats_V = torch.sum(feats * atten, dim=1)
        return feats_V


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8,
                 attention="se", dynamic=False):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        if dynamic:
            self.dkconvs = nn.ModuleList([])
            for i in range(self.nums):
                self.dkconvs.append(dkconv(width, kernel_size=kernel_size))
        else:
            convs = []
            bns = []
            num_pad = math.floor(kernel_size / 2) * dilation
            for i in range(self.nums):
                convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
                bns.append(nn.BatchNorm1d(width))
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        self.dynamic = dynamic
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        if attention == "se":
            self.atten = AttentionBlocks.SEModule(planes)
        elif attention == "spa":
            self.atten = AttentionBlocks.SPAModule(planes)
        elif attention == "eca":
            self.atten = AttentionBlocks.ECAModule()
        elif attention == "cbam":
            self.atten = AttentionBlocks.CBAModule(planes)
        else:
            raise ValueError('Undefined attention block.')

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            if self.dynamic:
                sp = self.dkconvs[i](sp)
            else:
                sp = self.convs[i](sp)
                sp = self.relu(sp)
                sp = self.bns[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.atten(out)
        out += residual
        return out

# BasicBlock 两个conv、bn层 -> attention模块


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention="SE", reduction=8):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 用于匹配 residual 和 out 的维度
        self.stride = stride
        if attention == "se":
            self.atten = AttentionBlocks.SEModule(planes, planes // reduction)
        elif attention == "spa":
            self.atten = AttentionBlocks.SPAModule(planes, planes // reduction)
        elif attention == "eca":
            self.atten = AttentionBlocks.ECAModule()
        elif attention == "cbam":
            self.atten = AttentionBlocks.CBAModule(planes, planes // reduction)
        else:
            raise ValueError('Undefined attention block.')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.atten(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
