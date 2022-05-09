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
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore
from Losses.label_smoothing_loss import LabelSmoothing
from Losses.cos_loss import CosLoss


def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction='sum')


def mean_square_error(logits, targets, **kwargs):
    return F.mse_loss(logits, 15*F.one_hot(targets, num_classes=kwargs['num_classes']).type(torch.FloatTensor).to(logits.device), reduction='sum')


def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)


def focal_loss_adaptive(logits, targets, **kwargs):
    return FocalLossAdaptive(gamma=kwargs['gamma'],
                             device=kwargs['device'])(logits, targets)


def label_smoothing(logits, targets, **kwargs):
    return LabelSmoothing(smoothing=kwargs['smoothing'])(logits, targets)

def cos_loss(logits, targets, **kwargs):
    return CosLoss()(logits, targets)


def mmce(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def mmce_weighted(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)
