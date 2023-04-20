"""
项目主代码

1、定义声明
2、加载模型参数
3、进行训练
"""

import argparse
import glob
import os
import torch
import warnings
import time

from torch.utils.data import DataLoader
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
from tools import *

# from ECAPAModel import ECAPAModel

# 主声明
parser = argparse.ArgumentParser(description="ECAPA_trainer")
# 训练设置
parser.add_argument('--num_frames',    type=int,    default=200,     help='音频帧的数量，200->2秒')
parser.add_argument('--max_epoch',     type=int,    default=80,      help='最大训练轮次')
parser.add_argument('--batch_size',    type=int,    default=400,     help='批数据大小')
parser.add_argument('--n_cpu',         type=int,    default=16,      help='数据载入cpu数')
parser.add_argument('--test_step',     type=int,    default=10,      help='test_step轮次后测试')
parser.add_argument('--lr',            type=float,  default=0.001,   help='学习率')
parser.add_argument('--lr_decay',      type=float,  default=0.97,    help='学习率衰减')

# 训练集、测试（数据、列表）路径和模型结果保存路径
parser.add_argument('--train_list',    type=str, default='',           help='训练集列表路径')
parser.add_argument('--train_path',    type=str, default='',           help='训练集数据路径')
parser.add_argument('--eval_list',     type=str, default='',           help='测试集列表路径')
parser.add_argument('--eval_path',     type=str, default='',           help='测试集书记路径')
parser.add_argument('--musan_path',    type=str, default='',           help='musan路径,用于数据增强')
parser.add_argument('--rir_path',      type=str, default='',           help='rir路径，用于数据增强')
parser.add_argument('--save_path',       type=str, default='',           help='模型、分数保存路径')
parser.add_argument('--initial_model', type=str, default='',        help='预训练模型')

# 模型以及损失函数设置
parser.add_argument('--channel',          type=int,     default=1024,     help='通道个数，即特征个数')
parser.add_argument('--margin',           type=float,   default=0.2,      help='AAM Softmax的损失裕度')
parser.add_argument('--scale',            type=float,   default=30,       help='AAM Softmax的缩放系数')
parser.add_argument('--n_class',          type=int,     default=5994,     help='说话人个数')

# 命令设置
parser.add_argument('--eval',       dest='eval',    action='store_true',    help='设置只做测试')

# 初始化
# 忽略警报性错误
warnings.simplefilter('ignore')
# cpu多进程共享策略，不知道具体内容
torch.multiprocessing.set_sharing_strategy('file_system')
# 声明, init_args 为tools库函数
args = parser.parse_args()
args = init_args(args)

# 载入数据， train_loader 为dataLoader库函数
# **vars(args) 将args 的参数转化为字典
# train_loader 为dataLoader 库定义的类
trainloader = train_loader(**vars(args))
trainLoader = DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_gpu, drop_last=True)

# 查找已存在的模型
modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
modelfiles.sort()

# 评估测试模式，使用已存在的模型， 位置为 args.initial_model
if args.eval:
    # ECAPAModel 为 ECAPAModel定义的类, 定义ECAPAModel 模型初始化以及训练过程
    s = ECAPAModel(**vars(args))
    print('Model %s loaded from previous state!' % args.initial_model)
    s.load_parameters(args.initial_model)
    EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
    print('EER %2.2f%% minDCF %.4f%%' % (EER, minDCF))
    quit()
