import torch
import torchaudio

import torch.nn as nn
import models.BasicBlocks as BasicBlocks

# ResNetSE 基准模型 与论文中结构有些许差别


class MainModel(nn.Module):
    def __init__(self, block=BasicBlocks.BasicBlock, layers=[3, 4, 6, 3], num_filters=[32, 64, 128, 256],
                 nOut=192, in_feat="fbank", in_dim=80, attention="SE", **kwargs):
        super(MainModel, self).__init__()
        print('Embedding size is %d, encoder ASP.' % nOut)
        self.specaug    = BasicBlocks.FbankAug()
        self.inplanes   = num_filters[0]
        self.in_feat    = in_feat
        self.in_dim     = in_dim
        # self.attention  = attention  # self.attention 在下面重新定义

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)  # (batch, num_filters[0]=32, in_dim ,time)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])  # (batch, num_filters[0]=32, in_dim ,time)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm   = nn.InstanceNorm1d(in_dim)
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

        outmap_size = int(in_dim/8)  # 仅适用fbank与mfcc, spectrogram有200维，取前[in_dim]维作为网络输入

        self.attention = nn.Sequential(  # SE attention
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        out_dim = num_filters[3] * outmap_size * 2
        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # layer1: (batch, inplanes=num_filters[0]=32, in_dim=80, time)
    # block = BasicBlocks.BasicBlock, inplanes = planes = 32, blocks = layers[0] = 3
    # blocks = 3 个 BasicBlock 块

    # layer[2,3,4]: inplanes = [32,64,128], planes = [64, 126, 256], blocks = [3, 6, 3], stride = [2, 2]
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # block.expansion = 1, 只作用于layer[2,3,4]
            downsample = nn.Sequential(  # downsample: 匹配 BasicBlock 中 residual 和 out 的维度
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),  # （batch, planes, in_dim, time)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # stride = 1 的卷积

        return nn.Sequential(*layers)

    def forward(self, x, aug):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if self.in_feat == "fbank":  # (batch, in_dim=80, time)
                    x = self.torchfb(x)+1e-6
                    x = x.log()
                    x = self.instancenorm(x)
                    if aug:
                        x = self.specaug(x)
                elif self.in_feat == "spectrogram":
                    x = self.torchspec(x)
                    # # 取前80维：基本覆盖人声频率范围
                    # x = x[:, 0:80, :]
                    x -= torch.mean(x, dim=-1, keepdim=True)  # 均值标准化
                    if aug:
                        x = self.specaug(x)
                elif self.in_feat == "mfcc":
                    x = self.torchmfcc(x)
                    mean = torch.mean(x, dim=-1, keepdim=True)
                    # # 差分特征
                    # delta_x = delta(x)
                    # delta_delta_x = delta(delta_x)
                    # x = torch.cat((x, delta_x, delta_delta_x), dim=-2)
                    # # std = torch.sqrt(torch.var(x,dim=-1,keepdim=True,unbiased=False))
                    # # x = (x-mean)/std  # cmvn 倒谱均值方差标准化
                    x -= mean  # 倒谱均值标准化
                    if aug:
                        x = self.specaug(x)
                else:
                    raise ValueError('Undefined input feature.')

                x = x.unsqueeze(1)  # (batch, 1, in_dim, time)

        x = self.conv1(x)  # (batch, num_filters[0]=32, in_dim ,time)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)  # (batch, num_filters[0]=32, in_dim ,time)
        x = self.layer2(x)  # (batch, num_filters[1]=64, in_dim/2 ,time/2)
        x = self.layer3(x)  # (batch, num_filters[2]=128, in_dim/4 ,time/4)
        x = self.layer4(x)  # (batch, num_filters[3]=256, in_dim/8 ,time/8)

        x = x.reshape(x.size()[0], -1, x.size()[-1])  # (batch, 256 * in_dim / 8, time / 8)
        w = self.attention(x)  # (batch, 256 * in_dim / 8, time / 8)
        mu = torch.sum(x * w, dim=2)  # (batch, 256 * in_dim /8)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)  # (batch, 256 * in_dim / 8 * 2)
        x = x.view(x.size()[0], -1)  # (batch, 256 * in_dim / 8 * 2)
        x = self.fc(x)  # (batch, nOut)
        return x
