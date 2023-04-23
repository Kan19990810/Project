"""
定义 ECAPA-TDNN 模型
"""

import math
import torch
import torchaudio

import torch.nn as nn
import torch.nn.functional as F


class PreEmphasis(torch.nn.Module):
    # 预加重函数
    # coef 预加重系数
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # register_buffer 创造一个新的变量缓冲区, 不会被优化器更新， 用于将非参数变量放到模型中
        self.register_buffer(
            # flipped_filter[-0.97, 1.]  在前面扩展了 2 维, shape : torch.Size([1, 1, 2])
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsequeeze(0)
        )

    def forward(self, inp: torch.tensor) -> torch.tensor:
        # inp.shape: torch.Size([batch, time]), length -> time
        # 训练时
        # (batch, 1, time)
        inp = inp.unsqueeze(1)
        # (batch, 1, time + 1), pad ：time 维度 + 1
        inp = F.pad(inp, (1, 0), 'reflect')
        # return （batch, 1, time)， 卷积核尺寸为 2 ，conv1d 后 time 维度 - 1
        return F.conv1d(inp, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_long_axis(self, x, dim):
        # 输入x 的维度 [batch, feature/channel, time]
        original_size = x.shape
        batch, fea, time = x.shape
        # 两种模式，feature_mask and time_mask
        if dim == 1:
            D = fea
            width_range = self.frea_mask_width
        else:
            D = time
            width_range = self.time_mask_width
        # mask_len 的 张量形状维(batch, 1, 1)
        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        # arange 的张量形状： (1, 1, D)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        # mask 的张量形状： (batch, 1, D)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        # mask 的张量形状变为(batch, D)
        mask = mask.any(dim=1)
        if dim == 1:
            # (batch, D, 1), 在feature 维度 mask
            mask = mask.unsqueeze(2)
        else:
            # (batch, 1, D)， 在time 维度 mask
            mask = mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        # 返回 初始维度 (batch, feature, time)
        return x.view(*original_size)

    def forward(self, x):

        x = self.mask_long_axis(x, dim=2)
        x = self.mask_long_axis(x, dim=1)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            # (batch, channel, time)
            # 在 channel 维度上 平均池化
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, inp):
        x = self.se(inp)
        return inp * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        # dense layer Conv1D + ReLU + BN
        # 将特征减小到 scale * width
        width      = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1   = nn.BatchNorm1d(width*scale)
        self.nums  = scale - 1
        convs      = []
        bns        = []

        # 进行空洞卷积 Res2 Dilated Conv1D + ReLU + BN
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ReLU()

        # Conv1D + ReLU + BN
        self.conv3 = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU()

        # SE-Block
        self.width = width
        self.se    = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        # 空洞卷积部分
        # 在channel 维， 根据 width 进行切分
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[0]
            else:
                sp = sp + spx[i]
            # 参数不共享，分开更新参数
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                # out 维数 (batch, channel - width, time)
                out = torch.cat((out, sp), 1)
            # torch.cat((out, spx[self.nums]), 1), 维数 (batch, channel, time)
            # 这样的用意是什么 🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️🤷‍♂️
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual

        return out


class ECAPA_TDNN(nn.Module):
    # ECAPA_TDNN模型
    # channel 通道个数
    def __init__(self, channel):
        super(ECAPA_TDNN, self).__init__()

        # Sequential 得到mel特征， sample_rate 采样率， n_fft FFT长度, win_length 窗口长度， hop_length 相邻两个滑动窗口帧之间的距离，
        #  f_min 最小音频频率，f_max 最大音频频率， window_fn 窗函数， n_mels 梅尔组件数
        self.torchfbank = torch.nn.Sequential(
            # PreEmphasis 预加重函数
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80)
        )
        # 对Fbank 特征进行 mask 加强
        self.specaug = FbankAug()

        # 对照论文 Conv1D + ReLU + BN(k=5, d=1), in:(batch, 80, time), out:(batch, channel, time)
        self.conv1 = nn.Conv1d(80, channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channel)
        # SE-Res2Block(k=3, d=2)
        self.layer1 = Bottle2neck(channel, channel, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channel, channel, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channel, channel, kernel_size=3, dilation=4, scale=8)
        # 多层特征聚合层 multi-layer feature aggregation MFA
        self.layer4 = nn.Conv1d(3*channel, 1536, kernel_size=1)
        # Self-Attention Pooling, SAP
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            # 在 time 维度上 Softmax , 每帧的注意力得分
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            # 在time 维度， 均值为0
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.specaug(x)
        # (batch, 80, time) -> (batch, channel = 1024, time)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        # (batch, 3 * channel = 3072, time) -> (batch, 1536, time)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        # (batch, 1536, time) -> (batch, 1536 * 3 = 4608, time)
        # 拼接 x, 均值， 标准差
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        # (barch, 1536 * 3 = 4608, time) -> (batch, 1536, time)
        w = self.attention(global_x)

        # Attentive Statistics Pooling, ASP
        # (batch, 1536, time) -> (batch, 3072)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)

        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        # return (batch, 192)
        return x
