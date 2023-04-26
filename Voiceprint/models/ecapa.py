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
