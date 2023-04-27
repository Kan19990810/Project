#!/bin/bash

#SBATCH -J voice                         # 作业名为 jobname
#SBATCH -o voice_1.out                           # 输出文件重定向到 test.out
#SBATCH -p debug                               # 作业提交的队列为 gpu
#SBATCH -N 1                                # 作业申请 1 个节点
#SBATCH -n 16                                 # 总计申请  个核心
#SBATCH --ntasks-per-node=16                #一个节点的核心数
#SBATCH -t 1:00:00                             # 任务运行的最长时间 小时
#SBATCH --gres=gpu:1                      # 单个节点使用 1 块 GPU 卡

#运行程序
#CUDA_VISIBLE_DEVICES=1
source /public/home/xiaok/xiaok/.bashrc
source activate Voiceold
python3 -u main.py