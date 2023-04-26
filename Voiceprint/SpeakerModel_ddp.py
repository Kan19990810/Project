import torch
import importlib
import sys
import time
import tqdm
import soundfile
import os
import numpy
import tools

import torch.nn as nn
import torch.nn.functional as F

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


class ModelTrainer(object):
    def __init__(self, speaker_model, rank, lr, test_step, lr_decay):
        self.model = speaker_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=test_step, gamma=lr_decay)
        self.rank = rank

    def train_network(self, epoch, loader, rank):
        self.model.train()
        index, top1, loss = 0, 0, 0
        lr = self.optimizer.param_groups[0]['lr']
        for num, (data, label) in enumerate(loader, start=1):
            self.model.zero_grad()
            label = torch.LongTensor(label).cuda()
            nloss, prec1 = self.model(data, label)
            nloss.backward()
            self.optimizer.step()
            index += len(label)
            top1 += prec1
            loss += nloss.detach().cpu().numpy()
            if rank == 0:
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                                 " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                                 " rank0 Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(label)))
                sys.stderr.flush()
        self.scheduler.step(epoch - 1)
        if rank == 0:
            sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(label)

    def eval_network(self, eval_list, eval_path, eval_frames, num_eval=5, **kwargs):
        self.model.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles), ncols =80):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            data1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
            max_audio = eval_frames * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=num_eval)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float)
            data2 = torch.FloatTensor(feats).cuda()

            with torch.no_grad():

                # compute score with data1 and data2
                embedding1 = self.model(data1)
                embedding1 = F.normalize(embedding1, p=2, dim=1)
                embedding2 = self.model(data2)
                embedding2 = F.normalize(embedding2, p=2, dim=1)
            embeddings[file] = [embedding1, embedding2]

        scores, labels = [], []

        for line in lines:
            # data1 & data2
            embedding11, embedding12 = embeddings[line.split()[1]]
            embedding21, embedding22 = embeddings[line.split()[2]]
            score1 = torch.mean(torch.matmul(embedding11, embedding21.T))
            score2 = torch.mean(torch.matmul(embedding21, embedding22.T))
            score = (score1 + score2)/2

            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        EER = tools.tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = tools.ComputeErrorRates(scores, labels)
        minDCF, _ = tools.ComputeMinDCF(fnrs, fprs, thresholds, 0.05, 1, 1)
        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.model.module.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.model.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.rank)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"
                      %(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)