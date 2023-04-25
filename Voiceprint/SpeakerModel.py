import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import sys
import time


class SpeakerNet1(nn.Module):
    def __init__(self, model, loss, nPerSpeaker, lr, lr_decay, test_step, **kwargs):
        super(SpeakerNet1, self).__init__()
        SpeakerEncoder = importlib.import_module("models." + model).__getattribute__("MainModel")
        SpeakerLoss = importlib.import_module("loss." + loss).__getattribute__("LossFunction")
        self.speaker_encoder = SpeakerEncoder(**kwargs).cuda()
        self.speaker_loss = SpeakerLoss(**kwargs).cuda()
        self.nPerSpeaker = nPerSpeaker
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=test_step, gamma=lr_decay)

    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optimizer.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            data = data.transpose(1, 0)
            self.zero_grad()
            data = data.reshape(-1, data.size()[-1]).cuda()
            labels = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            speaker_embedding = speaker_embedding.reshape(
                self.nPerSpeaker, -1, speaker_embedding.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optimizer.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " rank0 Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)