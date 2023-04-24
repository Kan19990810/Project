"""
训练模型以及测试模型
"""
import torch
import os
import sys
import tqdm
import numpy
import soundfile
import time
import pickle

import torch.nn as nn
import torch.nn.functional as F

from tools import *
from model import ECAPA_TDNN
from loss import AAMsoftmax


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, channel, n_class, margin, scale, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        # model模块中ECAPA_TDNN模型放入cuda中
        self.speaker_encoder = ECAPA_TDNN(channel=channel).cuda()
        # (batch, 192, time)
        self.speaker_loss = AAMsoftmax(n_class=n_class, margin=margin, scale=scale).cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime('%m-%d %H:%M:%S') + ' Model para number = %.2f million' % (
            sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}

        # 根据eval_list, 载入wav文件
        lines = open(eval_list).read().splitlines()
        # id10003 id10003/na8-QEFmj44/00003.wav
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        # tqdm 进度条提示模块
        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            # 将完全长度的音频存入 CUDA
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
            # 音频选择 300 帧长度
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            # 每个音频截取 num=5 段音频作为测试样本
            startframe = numpy.linespace(0, audio.shape[0] - max_audio, num=5)
            feats = []
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            # 将 num=5 段，长度=300 帧的音频存入cuda
            data_2 = torch.FloatTensor(feats).cuda()

            # 通过ECAPA 模型 得到嵌入向量
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                # (5, 192)
                embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []

        # 计算嵌入向量的得分
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # score_1  (1, 192) matmul (1, 192).T -> 1
            # score_2 (5, 192) matmul (5, 192).T -> (5, 5) -> 1
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # 计算EER 和 minDCF
        # tuneThresholdfromScore(),ComputeErrorRates(),ComputeMinDCF()为tools库函数内
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDCF(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self.state:
                name = name.replace('module,', '')
                if name not in self.state:
                    print('%s is not in the model.' % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print('Wrong parameter length: %s, model: %s, loaded %s' % (
                    origname, self.state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

    def train_network(self, epoch, loader):
        self.train()
        # 设置当前 epoch 对应的学习率
        self.scheduler.step(epoch - 1)

        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']

        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.datach().cpu().numpy()
            # sys.stderr 类似于中断？🤞🤞🤞🤞🤞
            sys.stderr.write(time.strftime('%m-%d %H:%M:%S') +
                             ' [%2d Lr: %5f, Training: %.2f%%, ' % (epoch, lr, 100 * (num / loader.__len__())) +
                             ' Loss: %.5f, ACC: %2.2f%% \r' % (loss / num, top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write('\n')
        return loss / num, lr, top1 / index * len(labels)
