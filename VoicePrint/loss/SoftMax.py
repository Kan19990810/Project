import torch
import tools

import torch.nn as nn

# SoftMax 损失函数


class LossFunction(nn.Module):
    def __init__(self, n_class, **kwargs):
        super(LossFunction, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.criterion      = nn.CrossEntropyLoss()
        self.fc             = nn.Linear(192, n_class)
        print('Initialised Softmax')

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == 192

        x = self.fc(x)
        nloss = self.criterion(x, label)
        prec1 = tools.accuracy(x.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1
