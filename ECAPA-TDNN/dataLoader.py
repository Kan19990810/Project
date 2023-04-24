"""
数据载入
"""

import glob
import numpy
import os
import random
import soundfile
import torch

from scipy import signal


class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames

        # 数据增强配置
        self.noisetypes = {'noise', 'speech', 'music'}
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}

        # 读取musan中的所有环境音频文件
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        # 分场景存放musan音频文件：speech、noise、music
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        # 读取rir中所有环境音频文件
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

        # 根据list文件加载数据和标签
        self.data_list = []
        self.data_label = []

        # 统计说话者，并按ID排序
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        # 构造说话者字典， key:id, ii:下标
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # 重新遍历list文件，构造训练数据集
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    # 根据data_list 加载音频数据，随机加载环境音频
    def __getitem__(self, index):

        # 读取wav文件， audio为音频， sr = 16000为采样率
        audio, sr = soundfile.read(self.data_list[index])

        # 根据设置帧率调整音频长度 帧移16ms = 160个数据点
        length = self.num_frames * 160 + 240
        # 如果音频长度短于预设长度，则填充
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        # 进行音频裁剪。 如果经过填充，从 0 开始，否则随机一个开始点
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame: start_frame + length]
        # 在audio 加上 0 维, 作用：让 audio 变成 1 * length 的矩阵
        audio = numpy.stack([audio], axis=0)

        # 数据增强，增加噪音
        augtype = random.randint(0, 5)
        if augtype == 0:  # 原音频
            audio = audio
        elif augtype == 1:  # 随机添加 rir 音频
            audio = self.add_rev(audio)
        elif augtype == 2:  # 演讲
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # 音乐
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # 噪音
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # 混合噪音
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        # 返回音频的 0 维 与标签， 作用：返回音频shape torch.Size([length])
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        # 在 rir 上增加 0 维 ， rir 的维数为 2 维 1 * n
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        # rir 标准归一化
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        # 返回 audio 与 rir 卷积结果，截取 与 audio 相同的长度, [:, :self.num_frames * 160 + 240] :返回 1 * length 矩阵
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        # clean_db 作用：音频分贝
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        # self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        numnoise = self.numnoise[noisecat]
        # self.noiselist: musan noise 文件表, 随选选取 numnoise 个音频文件
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame: start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)

            # noise_db: 噪音分贝
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            # self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            # 标准化 noiseaudio，根据信噪比进行改变噪声分贝
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        # 所有噪音音频在第 0 维拼接，再由 0 维进行累和。这样speech 有多个音频文件，累和会不会导致幅值过高？👍👍👍👍
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
