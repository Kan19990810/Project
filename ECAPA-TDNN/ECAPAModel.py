"""
训练模型以及测试模型
"""
import torch
import sys
import tqdm
import numpy
import soundfile
import time
import pickle

import torch.nn as nn

# from tools import *
# from loss import AAMsoftmax
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, channel, n_class, margin, scale, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        # model模块中ECAPA_TDNN模型放入cuda中
        self.speaker_encoder = ECAPA_TDNN(channel=channel).cuda()


