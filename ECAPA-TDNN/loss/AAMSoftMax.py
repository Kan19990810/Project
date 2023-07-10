import torch
import math
import tools

import torch.nn as nn
import torch.nn.functional as F

# AAMSoftMax 损失函数


class LossFunction(nn.Module):
    def __init__(self, n_class, m, s, **kwargs):
        super(LossFunction, self).__init__()

        # self.in_feats = out_dim
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True) ##维度随时修改!!!!
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        print('Initialised AAMSoftmax margin %.3f scale %.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == 192  # 维度随时修改!!!!
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = tools.accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
