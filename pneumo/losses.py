import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(inputs, targets):
    smooth = 1.

    iflat = F.sigmoid(inputs.view(-1))
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def focal_loss(inputs, targets, alpha=0.8, gamma=2, smooth=1):       
    #flatten label and prediction tensors
    inputs = F.sigmoid(inputs.view(-1))
    targets = targets.view(-1)

    #first compute binary cross-entropy 
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
    return focal_loss