import torch
import importlib

import torch.nn as nn


class SpeakerNet(nn.Module):
    def __init__(self, model, loss, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()
        SpeakerEncoder = importlib.import_module('models.' + model).__getattribute__('MainModel')
        SpeakerLoss = importlib.import_module('loss.' + loss).__getattribute__('LossFunction')
        self.speaker_encoder = SpeakerEncoder(**kwargs).cuda()
        self.speaker_loss = SpeakerLoss(**kwargs).cuda()
        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):
        data = data.reshape(-1, data.size()[-1].cuda)
        outp = self.speaker_encoder.forward(data, aug=False)
        if label is None:
            return outp
        else:
            outp = self.speaker_encoder.forward(data, aug=True)
            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.speaker_loss.forward(outp, label)
            return nloss, prec1
