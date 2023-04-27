import torch

import torch.nn as nn
import models.BasicBlocks as BasicBlocks


class MainModel(nn.Module):
    def __init__(self, nOut=1024, in_feat="fbank", in_dim=40, **kwargs):
        super(MainModel, self).__init__()
        print('Embedding size is %d, encoder ASP.' % nOut)
        self.feat = in_feat
        self.specaug = BasicBlocks.FbankAug()
        self.netcnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5, 7), stride=(1, 2), padding=(2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(4, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

        )

        self.encoder = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, nOut)
        self.instancenorm = nn.InstanceNorm1d(in_dim)
        if in_feat == "fbank":
            self.torchfb = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=in_dim)
                )
        elif in_feat == "spectrogram":
            self.torchspec = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=400, win_length=400, hop_length=160, window_fn=torch.hamming_window, normalized=True),
                )
        elif in_feat == "mfcc":
            self.torchmfcc = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=in_dim, log_mels=True, melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "f_min": 20, "f_max": 7600, "window_fn": torch.hamming_window}),
                )
        else:
            raise ValueError('Undefined input feature.')

    def forward(self, x, aug):
        with torch.no_grad():
            if self.feat == "fbank":
                x = self.torchfbank(x) + 1e-6
                x = x.log()
                # x = x - torch.mean(x, dim=-1, keepdim=True)
                # # 差分特征
                # delta_x = delta(x)
                # delta_delta_x = delta(delta_x)
                # x = torch.cat((x,delta_x,delta_delta_x),dim=-2)
                x = self.instancenorm(x).unsqueeze(1)
                if aug:
                    x = self.specaug(x)
            elif self.feat == "spectrogram":
                x = self.torchspec(x)
                # # 取前88维：基本覆盖人声频率范围
                # x = x[:, 0:88, :]
                x -= torch.mean(x, dim=-1, keepdim=True)  # 均值标准化
                if aug:
                    x = self.specaug(x)
            elif self.feat == "mfcc":
                x = self.torchmfcc(x)
                mean = torch.mean(x, dim=-1, keepdim=True)
                # # 差分特征
                # delta_x = delta(x)
                # delta_delta_x = delta(delta_x)
                # x = torch.cat((x, delta_x, delta_delta_x), dim=-2)
                # # std = torch.sqrt(torch.var(x,dim=-1,keepdim=True,unbiased=False))
                # # x = (x-mean)/std  #cmvn 倒谱均值方差标准化
                x -= mean  # 倒谱均值标准化
                if aug:
                    x = self.specaug(x)

            else:
                raise ValueError('Undefined input feature.')

        x = self.netcnn(x)
        x = self.encoder(x)
        x = x.view((x.size()[0], -1))
        x = self.fc(x)

        return x
