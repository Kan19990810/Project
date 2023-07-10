import glob
import os
import soundfile
import numpy
import random
import torch

import torch.distributed as dist

from scipy import signal
from torch.utils.data.sampler import Sampler


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(file_name, max_frames):
    audio, sr = soundfile.read(file_name)
    length = max_frames * 160 + 240
    if audio.shape[0] <= length:
        shortage = length - audio.shape[0]
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
    audio = audio[start_frame:start_frame + length]
    audio = numpy.stack([audio], axis=0)
    return audio


class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

        # Load data & labels
        self.label_dict = {}
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(index)
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, indices):
        feat = []
        for index in indices:
            # 读入语音并采
            audio = loadWAV(self.data_list[index], self.num_frames)
            # 数据增强 混响 加噪
            augtype = random.randint(0, 5)
            if augtype == 0:  # Original
                audio = audio
            elif augtype == 1:  # Reverberation
                audio = self.add_rev(audio)
            elif augtype == 2:  # Babble
                audio = self.add_noise(audio, 'speech')
            elif augtype == 3:  # Music
                audio = self.add_noise(audio, 'music')
            elif augtype == 4:  # Noise
                audio = self.add_noise(audio, 'noise')
            elif augtype == 5:  # Television noise
                audio = self.add_noise(audio, 'speech')
                audio = self.add_noise(audio, 'music')
            feat.append(audio)
        feat = numpy.concatenate(feat, axis=0)
        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class train_sampler(Sampler):
    def __init__(self, data_source, nPerSpeaker, utter_per_speaker, batch_size, ddp, seed, world_size, **kwargs):
        # super(train_sampler, self).__init__(data_source)
        # self.label_dict        = data_source.label_dict
        self.data_label = data_source.data_label
        self.nPerSpeaker = nPerSpeaker
        self.utter_per_speaker = utter_per_speaker
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.ddp = ddp
        self.world_size = world_size

    def __iter__(self):
        g = torch.Generator()  # g torch.Generator() 操作随机
        g.manual_seed(self.seed + self.epoch)  # manual_seed 手动随机种子
        indices = torch.randperm(len(self.data_label), generator=g).tolist()  # randperm 将序列随机打
        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)
        dictkeys = list(data_dict.keys())
        # dictkeys = list(self.label_dict.keys())
        dictkeys.sort()
        lol = lambda li, size: [li[i:i + size] for i in range(0, len(li), size)]  # 虚拟函数 返回[len(li) \\ size, size]

        flattened_list = []
        flattened_label = []
        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            # numSeg = k * nPerSpeaker
            numSeg = round_down(min(len(data), self.utter_per_speaker), self.nPerSpeaker)  # 向下取最大的可整除nPerSpeaker 的数
            # rp, (numSeg \\ nPerSpeaker = k, nPerSpeaker)
            rp = lol(numpy.arange(numSeg), self.nPerSpeaker)  # 返回以nPerSpeaker长度为间隔的顺序索引数组
            # rp = lol(numpy.random.permutation(len(data))[:numSeg], self.nPerSpeaker)
            # len(rp) = numSeg \\ nPerSpeaker, 相当于同时也有len(rp)个相同的
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                # indices [i: i + nPerSpeaker]
                # (numSeg \\ nPerSpeaker = k, nPerSpeaker)
                flattened_list.append([data[i] for i in indices])

        # Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        # mixid = numpy.random.permutation(len(flattened_label))
        mixlabel = []
        mixmap = []

        # Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            # startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        # nGPUs, device data to each GPU
        if self.ddp:
            total_size = round_down(len(mixed_list), self.batch_size * self.world_size)
            # 不明白🤦‍♂️🤦‍♂️🤦‍♂️🤦‍♂️🤦‍♂️🤦‍♂️🤦‍
            start_index = int((dist.get_rank()) / self.world_size * total_size)
            end_index = int((dist.get_rank() + 1) / self.world_size * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index: end_index])

        # 1GPU train
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])  # -> train_loader 的 __getitem__(self, indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
