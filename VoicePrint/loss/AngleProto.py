import numpy
import tools
import torch

import torch.nn as nn
import torch.nn.functional as F

# 角度原型损失函数 Angle Prototypical loss


class LossFunction(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.ce = torch.nn.CrossEntropyLoss()
        print('Initialised AngleProto')

    def forward(self, x, label=None):  # (batch, nPerSpeaker=2, 192)
        assert x.size()[1] >= 2
        out_anchor = torch.mean(x[:, 1:, :], 1)  # (batch, 192)
        out_positive = x[:, 0, :]  # (batch, 192)
        stepsize = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))  # (batch, 192, 1) and (1, 192, batch) -> (batch, batch)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()  # (batch)
        nloss = self.ce(cos_sim_matrix, label)
        prec1 = tools.accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1
