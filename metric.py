# coding=utf-8

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def metrics(y_scores, y_true, prefix='', do_reshape=True):
    if do_reshape:
        y_scores = y_scores.reshape(-1)
        y_true = y_true.reshape(-1)
    store_pr(y_scores, y_true, prefix)
    AUC(y_scores, y_true)
    P_at_N(y_scores, y_true)


def store_pr(y_scores, y_true, prefix):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    if prefix and prefix[-1] != '_':
        prefix = prefix + '_'
    np.save('plots/dsre/%sprecision.npy' % prefix, precision)
    np.save('plots/dsre/%srecall.npy' % prefix, recall)


def PR_Curve(y_scores, y_true, num=3000):
    y_predict = np.argmax(y_scores, axis=1)
    y_predict_score = np.max(y_scores, axis=1)
    n_target = (y_true.sum(axis=1) > 0).sum()
    n_predict = 0.
    n_right = 0.
    precisions = []
    recalls = []
    sorted_predict = np.argsort(y_predict_score)[::-1]
    for o in sorted_predict[:num]:
        n_predict += 1
        if y_true[o][y_predict[o]] == 1:
            n_right += 1
        precision = n_right / n_predict
        recall = n_right / n_target
        precisions.append(precision)
        recalls.append(recall)
    return np.array(recalls), np.array(precisions)


def AUC(y_scores, y_true):
    AUC_all = average_precision_score(y_true, y_scores)
    print('Area under the curve: {:.3}'.format(AUC_all))


def P_at_N(y_scores, y_true, numbers=list(range(100, 2001, 100))):
    sorted_predict = np.argsort(y_scores)[::-1]
    all_ps = []
    for num in numbers:
        top = sorted_predict[:num]
        correct_num = 0.0
        for o in top:
            if y_true[o] == 1:
                correct_num += 1.0
        p = correct_num / num
        all_ps.append(p)
        print('P@%d: ' % num, p)

    print('mean: ', sum(all_ps) / len(all_ps) if all_ps else 0.)
