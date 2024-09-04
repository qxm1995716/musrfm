# coding: utf-8
import torch.nn as nn
import torch
# here, the models mainly comes from the ResNet (He et al., 2016), where the official implement can be found on
# https://github.com/facebookarchive/fb.resnet.torch?tab=readme-ov-file, or you can find the pytorch implemention of 
# ResNet from timm lib. 


class ResBasicBlock(nn.Module):
    r"""
    This block does not consider any kinds of down sample operations.
    """
    expansion = 1

    def __init__(self,
                 in_channels,
                 channels,
                 ksize=3,
                 stride=1,
                 downsample=None,
                 channel_reduce_factor=1,
                 act_func=nn.ReLU,
                 norm_func=nn.BatchNorm2d,
                 drop_block=None,
                 drop_path=None,
                 ):
        super(ResBasicBlock, self).__init__()
        intermediate_channel = channels // channel_reduce_factor
        out_channels = channels * self.expansion
        # 基础模块
        self.conv1 = nn.Conv2d(in_channels, intermediate_channel, kernel_size=ksize, stride=stride,
                               padding=ksize // 2, bias=False)
        self.bn1 = norm_func(intermediate_channel)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_func(inplace=True)

        self.conv2 = nn.Conv2d(intermediate_channel, out_channels, kernel_size=ksize, stride=stride,
                               padding=ksize // 2, bias=False)
        self.bn2 = norm_func(out_channels)
        self.act2 = act_func(inplace=True)

        self.downsample = downsample
        self.drop_path = drop_path

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act2(x)

        return x
