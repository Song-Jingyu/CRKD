import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import LOSSES

__all__ = ["Affinity_Loss"]


class ConvChannel_monodistill(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_channel_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv_channel_layer(input)


"""
Standard Affinity_Loss
"""
@LOSSES.register_module()
class Affinity_Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0, downsample_size=[32, 16, 8, 4], input_channels=256, use_adapt=True):
        super(Affinity_Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.downsample_size = downsample_size
        self.use_adapt = use_adapt
        self.input_channels = input_channels
        if use_adapt:
            self.adapt_modules = nn.ModuleList()
            for i in range(len(downsample_size)):
                self.adapt_modules.append(ConvChannel_monodistill(input_channels, input_channels))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        feature_ditill_loss = 0.0
        B, C_in, H, W = input.shape
        B, C_ta, H, W = target.shape
        input_ds_lst = []
        target_ds_lst = []
        for i in range(len(self.downsample_size)):
            input_ds_temp = F.interpolate(input, size=self.downsample_size[i], mode="bilinear")
            input_ds_lst.append(input_ds_temp)
            target_ds_temp = F.interpolate(target, size=self.downsample_size[i], mode="bilinear")
            target_ds_lst.append(target_ds_temp)

            if self.use_adapt:
                input_ds_temp = self.adapt_modules[i](input_ds_temp)

            input_ds_temp = input_ds_temp.reshape(B, C_in, -1) # B, C_in, H*W
            input_affinity = torch.bmm(input_ds_temp.permute(0, 2, 1), input_ds_temp) # (B, H*W, C_in) * (B, C_in, H*W) = (B, H*W, H*W)

            target_ds_temp = target_ds_temp.reshape(B, C_ta, -1) # B, C_ta, H*W
            target_affinity = torch.bmm(target_ds_temp.permute(0, 2, 1), target_ds_temp) # (B, H*W, C_ta) * (B, C_ta, H*W) = (B, H*W, H*W)

            feature_ditill_loss += F.l1_loss(input_affinity, target_affinity, reduction=self.reduction) / B * self.loss_weight

        return feature_ditill_loss