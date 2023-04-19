"""
数据载入
"""

import glob, numpy, os, random, soundfile, torch
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
        # 在audio 加上 0 维
        audio = numpy.stack([audio], axis=0)
