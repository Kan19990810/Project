import loss.AngleProto as AngleProto
import loss.aamsoftmax as aamsoftmax
import torch.nn as nn


class LossFunction(nn.Module):

    def __init__(self, alpha=0.3, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.a = alpha
        self.aamsoftmax = aamsoftmax.LossFunction(**kwargs)
        self.angleproto = AngleProto.LossFunction(**kwargs)
        print('Initialised AAM-Softmax + A-Prototypical Loss')

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        nlossS, prec1   = self.aamsoftmax(x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))
        nlossP, _       = self.angleproto(x, None)
        loss = nlossS + self.a * nlossP
        return loss, prec1
