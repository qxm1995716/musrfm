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


# 轻量级的编码器，用于独立处理每个分辨率分支
class ResLWEncoder(nn.Module):
    def __init__(self, in_channels, ksize, stride, layers, block, act_fun=nn.ReLU, norm_func=nn.BatchNorm2d):
        super(ResLWEncoder, self).__init__()

        blocks = []

        for idx in range(len(layers)):
            channels = layers[idx]
            if in_channels != channels * block.expansion:
                downsample = nn.Sequential(*[
                    nn.Conv2d(in_channels, channels * block.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(channels * block.expansion)
                ])
            else:
                downsample = None

            blocks.append(block(in_channels, channels, ksize=ksize[idx], stride=stride, downsample=downsample,
                                act_func=act_fun, norm_func=norm_func))

            in_channels = channels * block.expansion

        self.net = nn.Sequential(*blocks)
        self.out_channel = in_channels

    def forward(self, x):

        fs = self.net(x)

        return fs


class TransUnit(nn.Module):
    def __init__(self, size, crop_radius=2, interp_method='bilinear'):
        super(TransUnit, self).__init__()

        self.interp_func = nn.Upsample(size=size, mode=interp_method, align_corners=True)

        self.crop_radius = crop_radius

    def forward(self, x):
        height, width = x.shape[-2], x.shape[-1]
        crop = x[:, :, height // 2 - self.crop_radius: height // 2 + self.crop_radius + 1,
                 width // 2 - self.crop_radius: width // 2 + self.crop_radius + 1]
        resize_fs = self.interp_func(crop)

        return resize_fs


class FusionUnit(nn.Module):
    def __init__(self, channel_h, channel_l, layers, ks, block_func):
        super(FusionUnit, self).__init__()

        self.channel_h = channel_h
        self.channel_l = channel_l
        init_channels = channel_h + channel_l
        # blocks
        blocks = []

        self.out_channel = layers[-1] * block_func.expansion

        for idx in range(len(layers)):
            channels = layers[idx] * block_func.expansion
            if init_channels != channels:
                downsample = nn.Sequential(*[
                    nn.Conv2d(init_channels, channels, kernel_size=1, padding=0, stride=1, bias=False),
                    nn.BatchNorm2d(channels)
                ])
            else:
                downsample = None

            blocks.append(block_func(init_channels, channels, ksize=ks[idx], downsample=downsample))

            init_channels = channels

        self.net = nn.Sequential(*blocks)

    def forward(self, h_x, l_x):
        x = torch.cat([h_x, l_x], dim=1)
        x = self.net(x)

        return x


class MSFPrototype(nn.Module):
    def __init__(self,
                 branch_channels,
                 encoder_layers,
                 encoder_ks,
                 block,
                 trans_layers,
                 trans_ks,
                 ):
        super(MSFPrototype, self).__init__()

        self.RBranch = ResLWEncoder(branch_channels, ksize=encoder_ks, stride=1, layers=encoder_layers, block=block)
        # self.R10Branch = ResLWEncoder(branch_channels, ksize=encoder_ks, stride=1, layers=encoder_layers, block=block)
        # self.R30Branch = ResLWEncoder(branch_channels, ksize=encoder_ks, stride=1, layers=encoder_layers, block=block)
        # self.R90Branch = ResLWEncoder(branch_channels, ksize=encoder_ks, stride=1, layers=encoder_layers, block=block)
        # self.R270Branch = ResLWEncoder(branch_channels, ksize=encoder_ks, stride=1, layers=encoder_layers, block=block)
        # self.R810Branch = ResLWEncoder(branch_channels, ksize=encoder_ks, stride=1, layers=encoder_layers, block=block)

        size = 15
        # 接在各分支后的裁剪及上采样分支
        self.Crop_UnSample = TransUnit(15, crop_radius=2, interp_method='bilinear')
        # 融合R810_UpSample与R270_Output的融合分支
        channels_l = encoder_layers[-1]
        channels_h = encoder_layers[-1]
        self.F_810_270 = FusionUnit(channel_h=channels_h, channel_l=channels_l, layers=trans_layers, ks=trans_ks,
                                    block_func=block)
        channels_l = trans_layers[-1]

        self.F_270_90 = FusionUnit(channel_h=encoder_layers[-1], channel_l=channels_l, layers=trans_layers, ks=trans_ks,
                                   block_func=block)
        self.F90_30 = FusionUnit(channel_h=encoder_layers[-1], channel_l=channels_l, layers=trans_layers, ks=trans_ks,
                                 block_func=block)
        self.F30_10 = FusionUnit(channel_h=encoder_layers[-1], channel_l=channels_l, layers=trans_layers, ks=trans_ks,
                                 block_func=block)

        # 输出分支
        self.Head = nn.Sequential(
            nn.Conv2d(channels_l, channels_l // 2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_l // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_l // 2, channels_l // 4, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_l // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_l // 4, 1, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        )

    def forward(self, x, add_gt=False, add_wl=False):

        r10 = x[:, :10, :, :]
        r30 = x[:, 10:20, :, :]
        r90 = x[:, 20:30, :, :]
        r270 = x[:, 30:40, :, :]
        r810 = x[:, 40:50, :, :]

        idxs = None

        if add_gt and add_wl:
            idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        elif not add_gt and add_wl:
            idxs = [0, 1, 2, 3, 4, 5, 6, 7, 9]

        elif add_gt and not add_wl:
            idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        else:
            idxs = [0, 1, 2, 3, 4, 5, 6, 7]

        r10 = r10[:, idxs, :, :]
        r30 = r30[:, idxs, :, :]
        r90 = r90[:, idxs, :, :]
        r270 = r270[:, idxs, :, :]
        r810 = r810[:, idxs, :, :]

        r10 = self.RBranch(r10)
        r30 = self.RBranch(r30)
        r90 = self.RBranch(r90)
        r270 = self.RBranch(r270)
        r810 = self.RBranch(r810)

        r810u = self.Crop_UnSample(r810)
        r270u = self.F_810_270(r270, r810u)
        r270u = self.Crop_UnSample(r270u)
        r90u = self.F_270_90(r90, r270u)
        r90u = self.Crop_UnSample(r90u)
        r30u = self.F90_30(r30, r90u)
        r30u = self.Crop_UnSample(r30u)
        r10u = self.F30_10(r10, r30u)

        output = self.Head(r10u)

        return output

