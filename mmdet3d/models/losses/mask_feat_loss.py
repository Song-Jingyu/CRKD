import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from mmdet3d.models.builder import LOSSES

__all__ = ["Mask_Feat_Loss", "Mask_Feat_Loss_Clip"]

"""
Standard Mask_Feat_Loss
"""
@LOSSES.register_module()
class Mask_Feat_Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(Mask_Feat_Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, gt_boxes: torch.Tensor):
        feature_ditill_loss = 0.0
        mask = self.calculate_box_mask(input, gt_boxes)

        input = input.permute(0, *range(2, len(input.shape)), 1)
        target = target.permute(0, *range(2, len(target.shape)), 1)
        batch_size = int(input.shape[0])
        positives = input.new_ones(*input.shape[:3])
        positives = positives * torch.any(target != 0, dim=-1)
        positives = positives * mask.cuda()

        reg_weights = copy.deepcopy(positives)
        pos_normalizer = torch.sum(positives)
        reg_weights /= pos_normalizer

        pos_inds = reg_weights > 0

        pos_input = input[pos_inds]
        pos_target = target[pos_inds]

        imitation_loss_src = self.compute_imitation_loss(pos_input,
                                                        pos_target,
                                                        weights=reg_weights[pos_inds])

        imitation_loss = imitation_loss_src.mean(-1)
        imitation_loss = imitation_loss.sum() / batch_size
        feature_ditill_loss += imitation_loss

        return feature_ditill_loss


    def calculate_box_mask(self, input, target):
        B, C, H, W = input.shape
        gt_mask = torch.zeros((B, H, W)).float()

        for i in range(B):
            for j in range(target[i].shape[0]):
                bbox2d_gt = target[i][j]

                left_top = bbox2d_gt[:2]
                right_bottom = bbox2d_gt[2:]

                left_top_x = int(left_top[0].item())
                left_top_y = int(left_top[1].item())

                right_bottom_x = int(right_bottom[0].item())
                right_bottom_y = int(right_bottom[1].item())

                gt_mask[i, left_top_y:right_bottom_y, right_bottom_x:left_top_x] = 1.0
        gt_mask = gt_mask.to(torch.int32)
        return gt_mask
    
    def compute_imitation_loss(self, input, target, weights):
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        loss = 0.5 * diff ** 2

        assert weights.shape == loss.shape[:-1]
        weights = weights.unsqueeze(-1)
        assert len(loss.shape) == len(weights.shape)
        loss = loss * weights

        return loss


@LOSSES.register_module()
class Mask_Feat_Loss_Clip(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(Mask_Feat_Loss_Clip, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, gt_boxes: torch.Tensor):
        feature_ditill_loss = 0.0
        mask = self.calculate_box_mask(input, gt_boxes)

        input = input.permute(0, *range(2, len(input.shape)), 1)
        target = target.permute(0, *range(2, len(target.shape)), 1)
        batch_size = int(input.shape[0])
        positives = input.new_ones(*input.shape[:3])
        positives = positives * torch.any(target != 0, dim=-1).float()
        positives = positives * mask.cuda()

        reg_weights = positives.float()
        pos_normalizer = positives.sum().float()
        reg_weights /= pos_normalizer

        pos_inds = reg_weights > 0
        pos_input = input[pos_inds].float()
        pos_target = target[pos_inds].float()

        imitation_loss_src = self.compute_imitation_loss(pos_input,
                                                        pos_target,
                                                        weights=reg_weights[pos_inds].float())  # [N, M]

        imitation_loss = imitation_loss_src.mean(-1)
        imitation_loss = imitation_loss.sum() / batch_size
        feature_ditill_loss += imitation_loss

        return feature_ditill_loss


    def calculate_box_mask(self, input, target):
        B, C, H, W = input.shape
        gt_mask = torch.zeros((B, H, W))
        for i in range(B):
            for j in range(target[i].shape[0]):
                bbox2d_gt = target[i][j]

                left_top = bbox2d_gt[:2]
                right_bottom = bbox2d_gt[2:]

                left_top_x = int(left_top[0].item())
                left_top_y = int(left_top[1].item())

                right_bottom_x = int(right_bottom[0].item())
                right_bottom_y = int(right_bottom[1].item())

                left_top_x = min(max(left_top_x, 0), 179)
                left_top_y = min(max(left_top_y, 0), 179)
                right_bottom_x = min(max(right_bottom_x, 0), 179)
                right_bottom_y = min(max(right_bottom_y, 0), 179)

                gt_mask[i, min([right_bottom_x, left_top_x]):max([right_bottom_x, left_top_x]), min([left_top_y, right_bottom_y]):max([left_top_y, right_bottom_y])] = 1.0
                
        return gt_mask
    
    def compute_imitation_loss(self, input, target, weights):
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        loss = 0.5 * diff ** 2

        assert weights.shape == loss.shape[:-1]
        weights = weights.unsqueeze(-1)
        assert len(loss.shape) == len(weights.shape)
        loss = loss * weights

        return loss