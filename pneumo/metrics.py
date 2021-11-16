
def dice_coeff(inputs, targets, threshold=0.5, eps=1e-3):
    
    iflat = (inputs.view(-1)> threshold).float()
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    if tflat.sum() == 0:
        return 1.
    else:
        return ((2. * intersection) / (iflat.sum() + tflat.sum()))