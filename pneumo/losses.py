import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(inputs, targets, eps=1):

    iflat = inputs.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + eps) / (iflat.sum() + tflat.sum() + eps))

def weighted_soft_dice_loss(inputs, targets, v2=0.9, eps=1e-4):
    """
    From https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9180275
    allows contribution of negative samples
    """
    v1 = 1 - v2

    iflat = inputs.view(-1)
    tflat = targets.view(-1)

    w = (tflat * (v2 - v1)) + v1
    g_iflat = w * (2 * iflat - 1)
    g_tflat = w * (2 * tflat - 1)
    intersection = (g_iflat * g_tflat).sum()

    calc = 1 - ((2 * intersection + eps)/ (torch.abs(g_iflat).sum() + torch.abs(g_tflat).sum() + eps))
    return torch.clip(calc, max=1-eps)

def focal_loss(inputs, targets, alpha=0.8, gamma=2):       
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    #first compute binary cross-entropy 
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
    return focal_loss