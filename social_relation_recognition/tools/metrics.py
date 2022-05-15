import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder


class ScoreMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch = 0
        self.step = 0
        self.value = 0.

    def update(self, epoch, step, value):
        self.epoch = epoch
        self.step = step
        self.value = value


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.
        self.val = 0.
        self.sum = 0.
        self.avg = 0.
        self.max = float("-inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if(self.val > self.max):
            self.max = self.val

        if(self.val < self.min):
            self.min = self.val


def multi_scores(pre_scores,
                 labels,
                 options=[
                     'accuracy',
                     'balanced_accuracy',
                     'mean_average_precision',
                     'class_recall',
                     'class_precision',
                     'class_average_precision',
                     'log_loss'
                 ]):

    pre_scores = np.nan_to_num(pre_scores)
    result = {}
    num_classes = pre_scores.shape[1]
    enc = OneHotEncoder()

    for op in options:

        if op == 'accuracy':
            scores = metrics.accuracy_score(
                labels,
                np.argmax(pre_scores, axis=1)
            )

        elif op == 'balanced_accuracy':
            scores = metrics.recall_score(
                labels,
                np.argmax(pre_scores, axis=1),
                labels=list(range(num_classes)),
                average='macro'
            )

        elif op == 'mean_average_precision':
            enc.fit(np.arange(num_classes).reshape(-1, 1))
            labels_oh = enc.transform(np.asarray(
                labels).reshape(-1, 1)).toarray()

            scores = metrics.average_precision_score(
                labels_oh,
                pre_scores,
                average='macro'
            )

        elif op == 'class_recall':
            scores = metrics.recall_score(
                labels,
                np.argmax(pre_scores, axis=1),
                labels=list(range(num_classes)),
                average=None
            )

        elif op == 'class_precision':
            scores = metrics.precision_score(
                labels,
                np.argmax(pre_scores, axis=1),
                labels=list(range(num_classes)),
                average=None
            )

        elif op == 'class_average_precision':
            enc.fit(np.arange(num_classes).reshape(-1, 1))
            labels_oh = enc.transform(np.asarray(
                labels).reshape(-1, 1)).toarray()

            scores = metrics.average_precision_score(
                labels_oh,
                pre_scores,
                average=None
            )

        elif op == 'log_loss':
            scores = metrics.log_loss(
                labels,
                pre_scores,
                labels=list(range(num_classes))
            )

        else:
            continue

        result[op] = scores

    return result
