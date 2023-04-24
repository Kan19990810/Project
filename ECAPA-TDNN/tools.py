"""
工具函数
"""

import numpy
import os

from operator import itemgetter
from sklearn import metrics


def init_args(args):
    # 分数保存路径
    args.score_save_path = os.path.join(args.save_path, 'score.txt')
    # 模型保存路径，并创建文件夹
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    # n_class 的topk 坐标 (batch, maxk = 1, time)
    _, pred = output.topk(maxk, 1, True, True)
    # (maxk = 1, batch, time)
    pred = pred.t()
    # 统计正确个数
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        # 对于 topk 分开统计正确个数
        correct_k = correct[:k].view(-1).floeat().sum(0, keepdims=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    # metrics.roc_curve 计算ROC曲线
    # 假阳率， 真阳率， 阈值
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # 假阴率 = 1 - 真阳率
    fnr = 1 - tpr
    tunedThreshold = []
    # target_fr 目标错误拒绝率
    if target_fr:
        for tfr in target_fr:
            # nanargmin 找到指定轴中最小值
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    # target_fa 目标错误接受率
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    # EER : 假阴率 = 假阳率
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    # tunedThreshold:最接近目标值的阈值、假阳、假阴
    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):
    #  zip 编成 index: threshold 字典， 按照index 排序
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fnrs[i - 1] + 1 - labels[i])
    # fnrs 错误拒绝,  fprs 错误接受
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    fnrs = [x / float(fnrs_norm) for x in fnrs]
    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds


def ComputeMinDCF(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float('inf')
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    # min_dcf = min_c_det / c_def 这个和论文表述的min-DCF 不一致 🤦‍♀️🤦‍♀️🤦‍♀️🤦‍♀️
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
