"""
Utilitary file that contains the BCEDicePenalizeBorderLoss loss

Taken from https://github.com/doodledood/carvana-image-masking-challenge/blob/master/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss2d(nn.Module):
    """ 
    Weighted version of the BCE loss, where each pixel contributes with a different weight
    """
    def __init__(self, **kwargs):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view(-1)
        t = labels.view(-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss


class WeightedSoftDiceLoss(nn.Module):
    """
    Weighted version of Soft Dice loss
    """
    def __init__(self, **kwargs):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = torch.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num,-1)
        w2    = w*w
        m1    = (probs  ).view(num,-1)
        m2    = (labels ).view(num,-1)
        intersection = (m1 * m2)
        smooth = 1.
        score = 2. * ((w2*intersection).sum(1)+smooth) / ((w2*m1).sum(1) + (w2*m2).sum(1)+smooth)
        score = 1 - score.sum()/num
        return score


class BCEDicePenalizeBorderLoss(nn.Module):
    """
    Mix between BCE and Soft Dice loss.
    O pixels around 1 pixels receive an extra attention with an increased weight.
    """
    def __init__(self, kernel_size=21, **kwargs):
        super(BCEDicePenalizeBorderLoss, self).__init__()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        self.kernel_size = kernel_size

    def to(self, device):
        super().to(device=device)
        self.bce.to(device=device)
        self.dice.to(device=device)

    def forward(self, logits, labels):
        a = F.avg_pool2d(labels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = torch.ones(a.size()).to(device=logits.device)

        w0 = weights.sum()
        weights = weights + ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0

        loss = self.bce(logits, labels, weights) + self.dice(logits, labels, weights)

        return loss







