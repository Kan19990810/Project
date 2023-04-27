
import torch.nn as nn
import models.BasicBlocks as BasicBlocks


class MainModel(nn.Module):
    def __init__(self, nOut=512, in_feat="fbank", in_dim=40, **kwargs):
        super(MainModel, self).__init__()
        print('Embedding size is %d, encoder TSP.' % nOut)
        self.feat = in_feat
        if in_feat == "fbank":
            self.torchfbank = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=in_dim),
            )
            conv_in = in_dim

        elif in_feat == "spectrogram":
            self.torchspec = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=400, win_length=400, hop_length=160, window_fn=torch.hamming_window, normalized=True),
                )
            conv_in = 200  # ? 201, 88, ……
            # spectrogram的维度不由input_dim控制 默认200: (n_fft/2)

        elif in_feat == "mfcc":
            self.torchmfcc = nn.Sequential(
                BasicBlocks.PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=in_dim, log_mels=True, melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "f_min": 20, "f_max": 7600, "window_fn": torch.hamming_window}),
                )
            conv_in = in_dim
        else:
            raise ValueError('Undefined input feature.')

        self.specaug = BasicBlocks.FbankAug()
        self.tdnn1 = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=512, kernel_size=5, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5)
        )
        self.tdnn2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5)
        )
        self.tdnn3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5)
        )
        self.tdnn4 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5)
        )
        self.tdnn5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(1500),
            nn.Dropout(p=0.5)
        )
        # TSP
        self.fc1   = nn.Linear(3000, 512)
        self.fc2   = nn.Linear(512,  512)