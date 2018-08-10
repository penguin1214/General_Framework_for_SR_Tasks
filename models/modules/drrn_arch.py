import torch
import torch.nn as nn
from math import sqrt

from .blocks import ConvBlock, DeconvBlock


class DRRN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recurs):
        super(DRRN, self).__init__()
        self.num_recurs = num_recurs

        self.conv0 = ConvBlock(in_channels, num_features, kernel_size=3, bias=False, act_type=None, norm_type=None)
        self.conv1 = ConvBlock(num_features, num_features, kernel_size=3, bias=False, act_type=None, norm_type=None)
        self.conv2 = ConvBlock(num_features, num_features, kernel_size=3, bias=False, act_type=None, norm_type=None)
        self.conv_out = ConvBlock(num_features, out_channels, kernel_size=3, bias=False, act_type=None, norm_type=None)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # TODO: why?

    def forward(self, x):
        residual = x
        input = self.conv0(self.relu(x))
        out = input

        for _ in range(self.num_recurs):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, input)

        out = self.conv_out(self.relu(out))
        out = torch.add(out, residual)
        return out
