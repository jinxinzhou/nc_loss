'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0, size_average='mean'):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logprobs  = F.log_softmax(input)
        nll_loss  = -logprobs.gather(1, target)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.size_average =='mean': return loss.mean()
        else: return loss.sum()