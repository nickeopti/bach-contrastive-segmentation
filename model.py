import torch.nn as nn


class Conv2dBlock:
    def __new__(cls, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        residual = self.block(x)
        return x + residual


class AttentionNetwork:
    conv2d_settings = [
        dict(kernel_size=3, stride=1, padding=pd, dilation=pd, bias=False)
        for pd in (1, 2, 3, 5, 10, 20)
    ]

    def __new__(cls, in_channels, internal_channels):
        first_conv_block = Conv2dBlock(in_channels, internal_channels, **cls.conv2d_settings[0])
        residual_blocks = [
            ResidualBlock(nn.Sequential(
                Conv2dBlock(internal_channels, internal_channels, **kw),
                Conv2dBlock(internal_channels, internal_channels, **kw)
            ))
            for kw in cls.conv2d_settings
        ]
        last_conv_block = nn.Conv2d(internal_channels, 1, kernel_size=1, stride=1, padding=0)

        return nn.Sequential(
            first_conv_block,
            *residual_blocks,
            last_conv_block,
            nn.Sigmoid()
        )


class TypeNetwork:
    def __new__(cls, in_channels):
        return nn.Sequential(
            Conv2dBlock(in_channels, 8, kernel_size=3),
            Conv2dBlock(8, 8, kernel_size=3, stride=2, dilation=2, bias=False),
            Conv2dBlock(8, 1, kernel_size=3, stride=2, dilation=2, bias=False),
            nn.Flatten(),
            nn.Linear(81, 32)
        )
