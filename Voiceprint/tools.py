import os
import numpy

from sklearn import metrics


def init_args(args):
    args.score_save_path = os.path.join(args.save_path, 'score.txt')
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedT = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedT.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedT.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    t = thresholds[idxE]
    eer = max(fpr[idxE], fnr[idxE]) * 100
    return tunedT, eer, fpr, fnr, t


def ComputeACC(scores, labels, t):
    pred = []
    for i in range(0, len(scores)):
        if scores[i]>t or scores[i]==t:
            pred.append(1)
        else:
            pred.append(0)
    acc = metrics.accuracy_score(labels, pred) * 100
    # f1score = metrics.f1_score(labels, pred, average='macro') * 100
    # 正负样本分布均匀且TP、TN均趋近于N/2 F1与acc趋于相等
    return acc


def ComputeErrorRates(scores, labels):
    sorted_indexes, thresholds = zip(*sorted([(index, threshold) for index, threshold in enumerate(scores)],key=itemgetter(1)))
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1-labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm
    fnrs = [x / float(fnrs_norm) for x in fnrs]
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


# 计算MinDCF
def ComputeMinDCF(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


# 计算acc
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
