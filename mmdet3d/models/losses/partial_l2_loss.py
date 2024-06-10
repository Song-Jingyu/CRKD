import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import LOSSES

__all__ = ["Partial_L2_Loss"]

"""
Standard Partial_L2_Loss
"""
@LOSSES.register_module()
class Partial_L2_Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(Partial_L2_Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, margin=None):
        if margin is not None: 
            target = torch.max(target, margin)
        loss = torch.nn.functional.mse_loss(input, target, reduction="none")
        loss = loss * ((input > target) | (target > 0)).float()
        loss = loss * self.loss_weight
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()