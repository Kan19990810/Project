import os


def init_args(args):
    args.score_save_path = os.path.join(args.save_path, 'score.txt')
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args
