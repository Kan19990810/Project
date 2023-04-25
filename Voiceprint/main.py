import argparse
import warnings
import torch
import tools
import sys

import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description="ASV_trainer")

parser.add_argument('--world_size',        type=int,   default=1,     help='Number of gpus')
parser.add_argument('--num_frames',        type=int,   default=200,   help='Duration of input segments, eg:200 for 2 second')
parser.add_argument('--max_epoch',         type=int,   default=1,     help='Maximum number of epochs')
parser.add_argument('--batch_size',        type=int,   default=400,   help='Batch size')
parser.add_argument('--utter_per_speaker', type=int,   default=500,   help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--n_cpu',             type=int,   default=16,    help='Number of data loader threads')
parser.add_argument('--test_step',         type=int,   default=1,     help='Test and save every [test_step] epochs')
parser.add_argument('--lr',                type=float, default=0.001, help='Learning rate')
parser.add_argument('--lr_decay',          type=float, default=0.97,  help='Learning rate decay every [test_step] epochs')
parser.add_argument('--eval_frames',       type=int,   default=300,   help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--seed',              type=int,   default=10,    help='Seed for the random number generator')

parser.add_argument('--train_list',    type=str, default="", help='The path of the training list')
parser.add_argument('--train_path',    type=str, default="", help='The path of the training data')
parser.add_argument('--eval_list',     type=str, default="", help='The path of the true evaluation list')
parser.add_argument('--eval_path',     type=str, default="", help='The path of the evaluation data')
parser.add_argument('--musan_path',    type=str, default="", help='The path of the MUSAN set')
parser.add_argument('--rir_path',      type=str, default="", help='The path of the RIR set')
parser.add_argument('--initial_model', type=str, default="", help='Path of initial model')
parser.add_argument('--save_path',     type=str, default="", help='Path to save score.txt and models')

parser.add_argument('--C',            type=int,   default=1024,         help='Channel size for speaker encoder')
parser.add_argument('--in_feat',      type=str,   default="fbank",      help='input features of speaker encoder')
parser.add_argument('--in_dim',       type=int,   default=64,           help='Number of mel filterbanks')
parser.add_argument('--attention',    type=str,   default="se",         help='attention module of speaker encoder')
parser.add_argument('--dynamic_mode', type=bool,  default=False,        help='dynamic mode of speaker encoder')
parser.add_argument('--model',        type=str,   default="ecapa",      help='Name of model definition')
parser.add_argument('--loss',         type=str,   default="aamsoftmax", help='Loss function')
parser.add_argument('--m',            type=float, default=0.2,          help='loss margin in AAM softmax')
parser.add_argument('--s',            type=float, default=30,           help='loss scale in AAM softmax')
parser.add_argument('--nPerSpeaker',  type=int,   default=1,            help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--n_class',      type=int,   default=5994,         help='Number of speakers')

parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--ddp',  dest='ddp',  action='store_true', help='Enable distributed training')

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = tools.init_args(args)


def main():
    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:', args.save_path)

    if args.ddp:
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
        print('************* Training on single GPU! *************')
        main_worker_single(0, args)


if __name__ == '__main__':
    main()
