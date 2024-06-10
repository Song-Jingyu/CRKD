from typing import List

import torch
from torch import nn
import numpy as np

from mmdet3d.models.builder import FUSERS

__all__ = ["GatedFuser"]


@FUSERS.register_module()
class GatedFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(GatedFuser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv_sensor_1 = nn.Conv2d(self.in_channels[0]+self.in_channels[1], self.in_channels[0], kernel_size=3, stride=1, padding=1)
        self.sigmoid_sensor_1 = nn.Sigmoid()
        self.input_conv_sensor_2 = nn.Conv2d(self.in_channels[0]+self.in_channels[1], self.in_channels[1], kernel_size=3, stride=1, padding=1)
        self.sigmoid_sensor_2 = nn.Sigmoid()

        self.output_fuser_conv_norm = nn.Sequential(
            nn.Conv2d(self.in_channels[0]+self.in_channels[1], out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.output_fuser_relu = nn.ReLU(True)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == 2
        input_feature = torch.cat((inputs[0], inputs[1]), dim=1)

        # Obtain the feature weight
        weight_sensor_1 = self.input_conv_sensor_1(input_feature)
        weight_sensor_1 = self.sigmoid_sensor_1(weight_sensor_1)
        weight_sensor_2 = self.input_conv_sensor_2(input_feature)
        weight_sensor_2 = self.sigmoid_sensor_2(weight_sensor_2)

        # Obatin the enhanced features
        feat_enh_sensor_1 = inputs[0] * weight_sensor_1
        feat_enh_sensor_2 = inputs[1] * weight_sensor_2

        # Output fuser
        output_feat = torch.cat((feat_enh_sensor_1, feat_enh_sensor_2), dim=1)
        output_feat_before_relu = self.output_fuser_conv_norm(output_feat)
        output_feat_after_relu = self.output_fuser_relu(output_feat_before_relu)
        return output_feat_before_relu, output_feat_after_relu, feat_enh_sensor_1, feat_enh_sensor_2

