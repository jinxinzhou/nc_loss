'''
This module contains methods for training models with different loss functions.
'''

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, mean_square_error, focal_loss, focal_loss_adaptive
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score
from Losses.loss import label_smoothing
from Losses.loss import cos_loss


loss_function_dict = {
    'cross_entropy': cross_entropy,
    'mean_square_error': mean_square_error,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'label_smoothing': label_smoothing,
    'cos_loss': cos_loss,
}

def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def print_and_save(text_str, file_stream):
    print(text_str)
    print(text_str, file=file_stream)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_single_epoch(epoch,
                       model,
                       train_loader,
                       optimizer,
                       device,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       smoothing=0.05,
                       loss_mean=False,
                       num_classes=100):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(data)
        if ('mmce' in loss_function):
            loss = (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, smoothing=smoothing, num_classes=num_classes, device=device))
        else:
            loss = loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, smoothing=smoothing, num_classes=num_classes, device=device)

        if loss_mean:
            loss = loss / len(data)

        prec1, prec5 = compute_accuracy(logits.detach().data, labels.detach().data, topk=(1, 5))

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)
        top1.update(prec1.item(), len(data))
        top5.update(prec5.item(), len(data))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    # print('====> [epoch: %d] | Loss: %.4f | top1: %.4f | top5: %.4f'.format(
    #     epoch, train_loss / num_samples, top1.avg, top5.avg))
    return train_loss / num_samples, top1.avg, top5.avg


def val_single_epoch(epoch,
                      model,
                      val_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0,
                      smoothing=0.05,
                     num_classes=100):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits, _ = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, smoothing=smoothing, num_classes=num_classes, device=device).item())
            else:
                loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, smoothing=smoothing, num_classes=num_classes, device=device).item()
            num_samples += len(data)

            prec1, prec5 = compute_accuracy(logits.detach().data, labels.detach().data, topk=(1, 5))
            top1.update(prec1.item(), len(data))
            top5.update(prec5.item(), len(data))

    # print('======> Test set loss: {:.4f}'.format(
    #     loss / num_samples))
    return loss / num_samples, top1.avg, top5.avg


def test_single_epoch(epoch,
                      model,
                      test_loader,
                      device,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0,
                      smoothing=0.05,
                      num_classes=100):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits, _ = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, smoothing=smoothing, num_classes=num_classes, device=device).item())
            else:
                loss += loss_function_dict[loss_function](logits, labels, gamma=gamma, lamda=lamda, smoothing=smoothing, num_classes=num_classes, device=device).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples
