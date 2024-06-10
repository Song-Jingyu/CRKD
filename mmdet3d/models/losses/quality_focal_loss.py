import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import LOSSES

__all__ = ["Quality_Focal_Loss", "Quality_Focal_Loss_no_reduction"]

"""
Standard Quality Focal Loss
"""
@LOSSES.register_module()
class Quality_Focal_Loss(nn.Module):
    '''
    input[B,M,C] not sigmoid 
    target[B,M,C]
    '''
    def __init__(self, beta = 2.0):

        super(Quality_Focal_Loss, self).__init__()
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor, pos_normalizer=torch.tensor(1.0)):

        scale_factor = input-target
        loss = F.binary_cross_entropy(input, target, reduction='none')*(scale_factor.abs().pow(self.beta))
        loss /= torch.clamp(pos_normalizer, min=1.0)
        return loss


"""
Quality Focal Loss with sigmoid input tensor
"""
@LOSSES.register_module()
class Quality_Focal_Loss_no_reduction(nn.Module):
    '''
    input[B,M,C] not sigmoid 
    target[B,M,C], sigmoid
    '''
    def __init__(self, beta = 2.0):

        super(Quality_Focal_Loss_no_reduction, self).__init__()
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor, pos_normalizer=torch.tensor(1.0)):

        pred_sigmoid = torch.sigmoid(input)
        scale_factor = pred_sigmoid-target
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')*(scale_factor.abs().pow(self.beta))
        loss /= torch.clamp(pos_normalizer, min=1.0)
        return loss