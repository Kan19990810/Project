import torchaudio
import torch

import torch.nn as nn

from models import BasicBlocks


class MainModel(nn.Module):

    def __init__(self, in_feat, in_dim, C, attention="se", dynamic_mode=False, **kwargs):
        super(MainModel, self).__init__()
        print('Embedding size is 192, encoder ASP.')
        print('Attention module is %s' % attention)
        self.feat = in_feat
        if in_feat == "fbank":
            self.torchfbank = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=in_dim),
            )
            conv_in = in_dim

        elif in_feat == "spectrogram":
            self.torchspec = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=400, win_length=400, hop_length=160,
                                                  window_fn=torch.hamming_window, normalized=True),
            )
            conv_in = 200
            # spectrogram的维度不由input_dim控制 默认200: (n_fft/2)

        elif in_feat == "mfcc":
            self.torchmfcc = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=in_dim, log_mels=True,
                                           melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "f_min": 20,
                                                      "f_max": 7600, "window_fn": torch.hamming_window}),
            )
            conv_in = in_dim
        else:
            raise ValueError('Undefined input feature.')

        self.specaug = BasicBlocks.FbankAug()
        # self.FBatten = frequencyBandAtten(64, r=4)

        self.conv1 = nn.Conv1d(conv_in, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = BasicBlocks.Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8, attention=attention, dynamic=dynamic_mode)
        self.layer2 = BasicBlocks.Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8, attention=attention, dynamic=dynamic_mode)
        self.layer3 = BasicBlocks.Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8, attention=attention, dynamic=dynamic_mode)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        # SAP/ASP
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        print('*****Initialised ECAPA-TDNN model with configuration:*****')
        print("input feat       = ", in_feat)
        print("input dim        = ", in_dim)
        print("channel          = ", C)
        print("attention module = ", attention)
        print("dynamic conv     = ", dynamic_mode)

    def forward(self, x, aug):
        with torch.no_grad():
            if self.feat == "fbank":
                x = self.torchfbank(x) + 1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                # # 差分特征
                # delta_x = delta(x)
                # delta_delta_x = delta(delta_x)
                # x = torch.cat((x,delta_x,delta_delta_x),dim=-2)
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

        # 频带注意力机制
        # x = self.FBatten(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        # ASP
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)  # output:(batch_size,3072)

        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
