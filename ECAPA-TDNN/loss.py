"""
AAMsoftmax 损失函数
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from tools import *


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, margin, scale):
        super(AAMsoftmax, self).__init__()
        self.margin = margin
        self.scale = scale
        # （out_feature, in_feature）
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        # 参数初始化为正态分布, 防止梯度爆炸
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        # 有其他用处，但没有在代码中体现
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, x, label=None):
        # x: (batch, 192, time)
        # cosine, sine, phi : (batch, n_class, time)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        # one_hot :根据label 进行填 1  （batch, n_class, time）
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # i = j -> one_hot * phi, i != j -> (1.0 - one_hot) * cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = self.ce(output, label)
        # accuracy tools库函数中的函数
        # detach 不会在优化中改变参数
        prec1 = accuracy(output.detach(), label.detach, topk=(1,))[0]

        return loss, prec1
