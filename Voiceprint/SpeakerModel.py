import torch
import importlib
import sys
import time
import tqdm
import os
import soundfile
import numpy
import tools

import torch.nn as nn
import torch.nn.functional as F


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
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) +
                             " rank0 Loss: %.5f, ACC: %2.2f%% \r" % (loss / num, top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles), ncols=80):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2

            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Coumpute EER and minDCF
        _, EER, _, _, t = tools.tuneThresholdfromScore(scores, labels, [1, 0.1])
        # fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        # minDCF, _ = ComputeMinDCF(fnrs, fprs, thresholds, 0.05, 1, 1)
        # acc = ComputeACC(scores, labels, t) ##训练时节省时间，注释掉
        minDCF, acc = 0, 0  # test时注释掉
        return EER, minDCF, acc

    def identification_network(self, enroll_list, iden_list, iden_path):
        self.eval()
        files = []
        database = torch.zeros((0,)).cuda()
        lines = open(enroll_list).read().splitlines()
        for line in lines:
            files.append(line)
        for idx, file in tqdm.tqdm(enumerate(files), total=len(files), ncols=80):
            audio, _ = soundfile.read(os.path.join(iden_path, file))
            # Full utterance
            data = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
            with torch.no_grad():
                emb = self.speaker_encoder.forward(data, aug=False)
                emb = F.normalize(emb, p=2, dim=1)
                database = torch.cat((database, emb), dim=0)
        print("voice feature database is set!")
        querys = []
        iden = torch.zeros((0,)).cuda()
        lines_q = open(iden_list).read().splitlines()
        for line in lines_q:
            querys.append(line)
        for idx, query in tqdm.tqdm(enumerate(querys), total=len(querys), ncols=80):
            audio, _ = soundfile.read(os.path.join(iden_path, query))
            data_q = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
            with torch.no_grad():
                emb_q = self.speaker_encoder.forward(data_q, aug=False)
                emb_q = F.normalize(emb_q, p=2, dim=1)
                iden = torch.cat((iden, emb_q), dim=0)
        print("speeches to be identified are encoded!")

        size = len(querys)
        score = torch.matmul(iden, database.T)
        score = score.detach().cpu()
        _, index1 = score.topk(1, dim=1, largest=True, sorted=False)
        top1, top3, top5 = 0, 0, 0
        for i in range(0, size):
            if index1[i][0] == i // 30:
                top1 += 1
        _, index3 = score.topk(3, dim=1, largest=True, sorted=False)
        for i in range(0, size):
            for j in range(0, 3):
                if index3[i][j] == i // 30:
                    top3 += 1
                    break
        _, index5 = score.topk(5, dim=1, largest=True, sorted=False)
        for i in range(0, size):
            for j in range(0, 5):
                if index5[i][j] == i // 30:
                    top5 += 1
                    break
        return top1 * 100 / size, top3 * 100 / size, top5 * 100 / size

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
