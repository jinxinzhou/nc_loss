'''
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + MMCE_weighted
4. Cross Entropy + MMCE
5. Brier Score
'''

import torch
from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.label_smoothing_loss import LabelSmoothing


def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction=kwargs['loss_mean'])


def mean_square_error(logits, targets, **kwargs):
    return F.mse_loss(logits, kwargs['rescale']*F.one_hot(targets, num_classes=kwargs['num_classes']).type(torch.FloatTensor).to(logits.device), reduction=kwargs['loss_mean'])


def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'], size_average=kwargs['loss_mean'])(logits, targets)


def label_smoothing(logits, targets, **kwargs):
    return LabelSmoothing(smoothing=kwargs['smoothing'], size_average=kwargs['loss_mean'])(logits, targets)


