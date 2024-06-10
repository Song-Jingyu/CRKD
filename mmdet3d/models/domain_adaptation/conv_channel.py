import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import DAS

__all__ = ["ConvChannel"]

@DAS.register_module()
class ConvChannel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_channel_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv_channel_layer(input)