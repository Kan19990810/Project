import argparse
import warnings
import torch
import tools
import sys
import dataLoader
import glob
import os
import time
import tensorboardX

from SpeakerModel import SpeakerNet1
from SpeakerModel_ddp import SpeakerNet
from SpeakerModel_ddp import ModelTrainer
from torch.utils.data.dataloader import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist

parser = argparse.ArgumentParser(description="ASV_trainer")

# 训练设置
parser.add_argument('--world_size', type=int, default=1, help='GPU数量')
parser.add_argument('--num_frames', type=int, default=200, help='输入音频帧数： 200 -> 2秒')
parser.add_argument('--max_epoch', type=int, default=100, help='最大训练轮数')
parser.add_argument('--batch_size', type=int, default=400, help='批数据大小')
parser.add_argument('--utter_per_speaker', type=int, default=500, help='每个轮次中每个说话人的最大音频数')
parser.add_argument('--n_cpu', type=int, default=16, help='数据加载CPU数量')
parser.add_argument('--test_step', type=int, default=1, help='test_step轮次后测试')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--lr_decay', type=float, default=0.97, help='学习率衰减')
parser.add_argument('--eval_frames', type=int, default=300, help='测试输入音频帧数')
parser.add_argument('--seed', type=int, default=10, help='随机种子')

# 训练、测试的列表、数据保存路径
parser.add_argument('--train_list', type=str, default="lists/vox2_train.txt", help='训练列表路径')
parser.add_argument('--train_path', type=str, default="/public/home/xiaok/xiaok/data/voxceleb2", help='训练数据路径')
parser.add_argument('--eval_list', type=str, default="lists/vox1_O.txt", help='测试列表路径')
parser.add_argument('--eval_path', type=str, default="/public/home/xiaok/xiaok/data/voxceleb1", help='测试数据路径')
parser.add_argument('--musan_path', type=str, default="/public/home/xiaok/xiaok/data/musan_split", help='MUSAN数据路径')
parser.add_argument('--rir_path', type=str, default="/public/home/xiaok/xiaok/data/RIRS_NOISES/simulated_rirs", help='RIR数据路径')
parser.add_argument('--initial_model', type=str, default="", help='预训练模型')
parser.add_argument('--save_path', type=str, default="exps/exp1", help='模型、分数保存路径')

# 模型设置
parser.add_argument('--C', type=int, default=1024, help='通道个数，即特征个数')
parser.add_argument('--in_feat', type=str, default="fbank", help='模型的输入特征模式')
parser.add_argument('--in_dim', type=int, default=64, help='模型的输入特征个数')
parser.add_argument('--attention', type=str, default="se", help='模型的注意力模式')
parser.add_argument('--dynamic_mode', type=bool, default=False, help='动态卷积模式')
parser.add_argument('--model', type=str, default="ECAPA", help='模型名字')
parser.add_argument('--loss', type=str, default="AAMSoftMax", help='损失函数')
parser.add_argument('--m', type=float, default=0.2, help='AAM Softmax的损失裕度')
parser.add_argument('--s', type=float, default=30, help='AAM Softmax的缩放系数')
parser.add_argument('--nPerSpeaker', type=int, default=1, help='每批次每个说话人的语音数， 用于度量学习')
parser.add_argument('--n_class', type=int, default=5994, help='说话人个数')

# 模式设置
parser.add_argument('--eval', dest='eval', action='store_true', help='测试模式')
parser.add_argument('--ddp', dest='ddp', action='store_true', help='分布式训练')

# 初始化
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = tools.init_args(args)


# 多卡训练函数
def main_worker(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"
    rank1 = rank
    torch.cuda.set_device(rank1)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    dist.barrier()
    device = torch.device(f'cuda:{rank1}')
    print("Using device:{}\n".format(device))
    s = SpeakerNet(**vars(args)).to(device)
    # s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(s).to(device)  # 同步BN
    s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[rank1])

    train_dataset = dataLoader.train_loader(**vars(args))
    train_data_sampler = dataLoader.train_sampler(train_dataset, **vars(args))
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.n_cpu,
                                   sampler=train_data_sampler,
                                   pin_memory=False,
                                   worker_init_fn=dataLoader.worker_init_fn,
                                   drop_last=True,)

    trainer = ModelTrainer(s, rank=rank, lr=args.lr, test_step=args.test_step, lr_decay=args.lr_decay)

    if args.eval and rank != 0:
        quit()
    if args.eval and rank == 0:
        print("----------- eval mode: start evaluation! ----------")
        print("Model %s loaded from previous state!" % args.initial_model)
        trainer.load_parameters(args.initial_model)
        EER, minDCF = trainer.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, eval_frames=args.eval_frames)
        print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
        quit()

    modelfiles = glob.glob('%s/model_0*.pt' % args.model_save_path)
    modelfiles.sort()
    checkpoint_path = os.path.join(args.model_save_path, "initial_weights.pt")
    if rank == 0:
        trainer.save_parameters(checkpoint_path)
    dist.barrier()

    if len(modelfiles) >= 1:
        trainer.load_parameters(modelfiles[-1])
        print("----------- Model %s loaded from previous state! Start training! -----------" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1

    else:
        if rank == 0:
            print("----------- Using initialized weights! Start training! ----------")
        trainer.load_parameters(checkpoint_path)
        epoch = 1

    if rank == 0:
        print('Loaded the ddpmodel on GPU')
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in s.parameters()) / 1024 / 1024))

    EERs = []
    score_file = open(args.score_save_path, "a+")

    # tensorboard
    writer = tensorboardX.SummaryWriter('logs/exp2')

    while epoch <= args.max_epoch:

        # 主训练代码
        train_data_sampler.set_epoch(epoch)
        loss, lr, acc = trainer.train_network(epoch=epoch, loader=train_data_loader, rank=rank)
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('acc', acc, global_step=epoch)

        if rank == 0 and epoch % args.test_step == 0:
            # 保存参数
            trainer.save_parameters(args.model_save_path + "/model_%04d.pt" % epoch)

            # 验证，计算eer、acc、minDCF， 保存结果至score.txt
            EERs.append(
                trainer.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, eval_frames=args.eval_frames)[
                    0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%" % (epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" % (
                epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()
        epoch += 1

    writer.close()
    score_file.close()

    if rank == 0:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    dist.destroy_process_group()


# 单卡训练函数
def main_worker_single(gpu, args):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    s = SpeakerNet1(**vars(args)).to(device)
    print("Using device:{}\n".format(device))
    train_dataset = dataLoader.train_loader(**vars(args))
    train_data_sampler = dataLoader.train_sampler(train_dataset, **vars(args))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_cpu,
                                   sampler=train_data_sampler, pin_memory=False,
                                   worker_init_fn=dataLoader.worker_init_fn,
                                   drop_last=True, )

    modelfiles = glob.glob('%s/model_0*.pt' % args.model_save_path)
    modelfiles.sort()
    if args.eval:
        print("Model %s loaded from previous state!" % args.initial_model)
        s.load_parameters(args.initial_model)
        EER, minDCF, _ = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
        print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
        quit()
    if args.initial_model != "":
        print("Model %s loaded from previous state!" % args.initial_model)
        s.load_parameters(args.initial_model)
        epoch = 1
    elif len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s.load_parameters(modelfiles[-1])
    else:
        epoch = 1

    print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
            sum(param.numel() for param in s.parameters()) / 1024 / 1024))
    EERs = []
    score_file = open(args.score_save_path, "a+")
    # train_loss = open(os.path.join(args.save_path, 'train_loss.txt'), "a+")
    # eer_file = open(os.path.join(args.save_path, 'eer.txt'), "a+")
    # eval_loss = open(os.path.join(args.save_path, 'eval_loss.txt'), "a+")

    while 1:
        # Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=train_data_loader)
        # loss1 = s.compute_eval_loss(loader=eval_data_loader)  # Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            s.save_parameters(args.model_save_path + "/model_%04d.pt" % epoch)
            EERs.append(s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%" % (epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" % (
                             epoch, lr, loss, acc, EERs[-1], min(EERs)))

            # train_loss.write("%f, " % loss)  # 与score_file填写内容有重复，用于观察loss、eer走势
            # eer_file.write("%2.2f, " % (EERs[-1]))

            score_file.flush()
            # train_loss.flush()
            # eer_file.flush()

            # eval_loss.write("%f, "%(loss1))  # Evaluation every [test_step] epochs
            # eval_loss.flush()  # Evaluation every [test_step] epochs

        if epoch >= args.max_epoch:
            quit()

        epoch += 1

    score_file.close()


def main():
    # print('Python Version:', sys.version)
    # print('PyTorch Version:', torch.__version__)
    # print('Number of GPUs:', torch.cuda.device_count())
    # print('Save path:', args.save_path)

    if args.ddp:
        # 多卡训练
        print('************* Training on multi-GPUs! *************')
        processes = []
        mp.set_start_method("spawn")
        for rank in range(args.world_size):
            p = mp.Process(target=main_worker, args=(rank, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        # 单卡训练
        print('************* Training on single GPU! *************')
        main_worker_single(0, args)


if __name__ == '__main__':
    main()
