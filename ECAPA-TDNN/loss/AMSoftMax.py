import torch
import tools

import torch.nn as nn

# AMSoftMax 损失函数


class LossFunction(nn.Module):
    def __init__(self, n_class, m, s, **kwargs):
        super(LossFunction, self).__init__()

        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        print('Initialised AMSoftmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == 192

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)  # (batch, 1)
        x_norm = torch.div(x, x_norm)  # (batch ,192)
        w_norm = torch.norm(self.weight, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.weight, w_norm)
        costh = torch.mm(x_norm, w_norm)  # (batch, batch)
        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        prec1 = tools.accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
