"""
工具函数
"""

import os, numpy, torch

# from sklearn import metrics
# from operator import itemgetter
# import torch.nn.functional as F

def init_args(args):
    # 分数保存路径
    args.score_save_path = os.path.join(args.save_path, 'score.txt')
    # 模型保存路径，并创建文件夹
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args
