import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups: int = 1,
        BN: bool = True,
        Act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.bias = False if BN else True
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=self.bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = Act

    def forward(self, x):
        x = self.conv(x)
        if not self.bias:
            x = self.bn(x)
        return self.relu(x)


class ResNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ConvBlock(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            ConvBlock(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                Act=nn.Identity(),
            ),
        )
        self.identity = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            Act=nn.Identity(),
        )
        if stride == 1 and in_channels == out_channels:
            self.identity = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.identity(x) + self.conv(x))
