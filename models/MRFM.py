# coding: utf-8
import torch
import torch.nn as nn
from models.BasicUnits import ResBasicBlock
import torch.nn.functional as F


def central_crop_resize_func(input, radius, size):
    height, width = input.shape[2], input.shape[3]
    crop_tensor = input[:, :, height // 2 - radius: height // 2 + radius + 1,
                  width // 2 - radius: width // 2 + radius + 1]
    crop_tensor = F.interpolate(crop_tensor, size=size, mode='bilinear')

    return crop_tensor


class ResBranch(nn.Module):
    def __init__(self, in_channel, channel, layers_num, ks, block_func=ResBasicBlock, act_func=nn.ReLU,
                 norm_func=nn.BatchNorm2d):
        super(ResBranch, self).__init__()

        blocks = []

        down_sample = None

        for idx in range(layers_num):

            if in_channel != channel * block_func.expansion:

                down_sample = nn.Sequential(
                    nn.Conv2d(in_channel, channel * block_func.expansion, kernel_size=1, stride=1, padding=0,
                              bias=False),
                    nn.BatchNorm2d(channel * block_func.expansion)
                )
            else:
                down_sample = None

            blocks.append(ResBasicBlock(in_channel, channel * block_func.expansion, ksize=ks, stride=1,
                                        downsample=down_sample, act_func=act_func, norm_func=norm_func))

            in_channel = channel * block_func.expansion

        self.model = nn.Sequential(*blocks)
        self.out_channel = in_channel

    def forward(self, x):
        x = self.model(x)
        return x


class MRB(nn.Module):
    def __init__(self, branch_in_channels, branch_out_channels, branch_layers, block_func, ks,
                 resolutions=[1, 3, 9, 27, 81], act_func=nn.ReLU, norm_func=nn.BatchNorm2d):
        super(MRB, self).__init__()
        # 构建总的分支
        self.branch_in_channels = branch_in_channels
        self.branch_out_channels = [branch_out_channels[idx] * block_func.expansion for idx in
                                    range(len(branch_out_channels))]
        self.branch_layers = branch_layers
        self.act_func = act_func
        self.norm_func = norm_func
        self.ks = ks
        self.resolutions = resolutions

        # branch
        self.branch = self._make_branches(branch_layers, block_func, branch_in_channels,
                                          branch_out_channels)

        # 构建融合模块 upward_branches
        self.upward_cb, self.upward_fb = self._make_upward_processor(branch_out_channels, blocks_num=1)

    def _make_branches(self, branch_layers, block_func, branch_in_channels, branch_out_channels):
        branches = []

        for idx in range(len(self.resolutions)):
            in_c = branch_in_channels[idx]
            out_c = branch_out_channels[idx]
            nums_count = branch_layers[idx]

            branches.append(ResBranch(in_c, out_c, nums_count, self.ks, block_func=block_func, act_func=self.act_func,
                                      norm_func=self.norm_func))

        return nn.ModuleList(branches)

    def _make_upward_processor(self, branch_channels, blocks_num=1):
        upward_convs = []
        fusion_convs = []
        # 构建多个卷积，对应每一条分辨率分支
        for idx in range(len(self.resolutions)):
            # ---------------------------------------------------------------------
            block = []
            for iidx in range(blocks_num):
                block.append(
                    ResBasicBlock(branch_channels[idx], branch_channels[idx], ksize=3, stride=1, downsample=None,
                                  act_func=self.act_func, norm_func=self.norm_func))
            upward_convs.append(nn.Sequential(*block))

        # 构建后续处理模块
        # 该模块从次底层开始向高分辨率分支计算
        for idx in range(len(self.resolutions)):
            if idx == len(self.resolutions) - 1:
                fusion_convs.append(nn.Identity())
            else:
                fuse_unit = []

                downsample = nn.Sequential(
                    nn.Conv2d(branch_channels[idx] + branch_channels[idx + 1], branch_channels[idx], kernel_size=1,
                              stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(branch_channels[idx])
                )
                fuse_unit.append(ResBasicBlock(branch_channels[idx] + branch_channels[idx + 1], branch_channels[idx],
                                               ksize=3, stride=1, downsample=downsample, act_func=self.act_func,
                                               norm_func=self.norm_func))

                for iidx in range(blocks_num - 1):
                    fuse_unit.append(ResBasicBlock(branch_channels[idx], branch_channels[idx], ksize=3, stride=1,
                                                   downsample=None, act_func=self.act_func, norm_func=self.norm_func))

                fusion_convs.append(nn.Sequential(*fuse_unit))

        upward_convs = nn.ModuleList(upward_convs)
        fusion_convs = nn.ModuleList(fusion_convs)

        return upward_convs, fusion_convs

    def forward(self, r10, r30, r90, r270, r810):
        height, width = r10.shape[2], r10.shape[3]
        radius = (height // 3) // 2

        # 对不同分辨率特征图进行处理
        r10 = self.branch[0](r10)
        r30 = self.branch[1](r30)
        r90 = self.branch[2](r90)
        r270 = self.branch[3](r270)
        r810 = self.branch[4](r810)

        # 对各个分支进行更新，自下而上
        r810 = self.upward_fb[-1](self.upward_cb[-1](r810))
        # r270分支接收r810分支[upward_conv后]的crop_resize特征图，与r270分支特征图concat后进入fb[-2]，融合两条分支后得到r270融合特征
        r270_a = central_crop_resize_func(r810, radius, [height, width])
        r270_c = self.upward_cb[-2](r270)
        r270 = self.upward_fb[-2](torch.cat([r270_c, r270_a], dim=1))
        # r90分支接收r270分支[upward_conv后]的crop_resize特征图，与r90分支特征图concat后进入fb[-3]，融合两条分之后得到r90融合特征
        r90_a = central_crop_resize_func(r270, radius, [height, width])
        r90_c = self.upward_cb[-3](r90)
        r90 = self.upward_fb[-3](torch.cat([r90_c, r90_a], dim=1))
        # r30分支接收r90分支[upward_conv后]的crop_resize特征图，与r30分支特征图concat后进入fb[-4]，融合两条分支后得到r30融合特征
        r30_a = central_crop_resize_func(r90, radius, [height, width])
        r30_c = self.upward_cb[-4](r30)
        r30 = self.upward_fb[-4](torch.cat([r30_c, r30_a], dim=1))
        # r10分支接收r30分支[upward_conv后]的crop_resize特征图，与r10分支特征图concat后进入fb[-5]，融合两条分支后得到r10融合特征
        r10_a = central_crop_resize_func(r30, radius, [height, width])
        r10_c = self.upward_cb[-5](r10)
        r10 = self.upward_fb[-5](torch.cat([r10_c, r10_a], dim=1))

        return r10, r30, r90, r270, r810


class MRNet(nn.Module):
    def __init__(self,
                 in_channels,
                 branch_channels,  # list，每个element则是一个list，对应了每个分支的通道数量
                 branch_layers,  # list, 每个element则同样是一个list，其对应了每个分支中模块的数量
                 kernel_size,  # list or int，如果是list则说明每个stage的卷积核大小不一致
                 block_func,  # 卷积模块，此处实际上固定为了ResBasicBlock
                 resolution=[1, 3, 9, 27, 81],  # 每个分支的分辨率/10，因为是sentinel-2数据所有除以10，
                 act_func=nn.ReLU,
                 norm_func=nn.BatchNorm2d,
                 head_block_nums=3
                 ):
        super(MRNet, self).__init__()

        # 由MBR构建各个分支
        # 预处理头处理过后的各个分支，其通道数量是一致的
        branch_input_channels = [in_channels for idx in range(len(resolution))]
        if type(kernel_size) == list:
            ks_1 = kernel_size[0]
            ks_2 = kernel_size[1]
            ks_3 = kernel_size[2]
            ks_4 = kernel_size[3]
        else:
            ks_1 = kernel_size
            ks_2 = kernel_size
            ks_3 = kernel_size
            ks_4 = kernel_size

        # STAGE 1
        f_c = 0
        self.MRB_Stage_1 = MRB(branch_input_channels, branch_channels[0], branch_layers[0], block_func, ks_1,
                               resolutions=resolution, act_func=act_func, norm_func=norm_func)
        branch_input_channels = self.MRB_Stage_1.branch_out_channels
        f_c = f_c + branch_input_channels[0]

        # STAGE 2
        self.MRB_Stage_2 = MRB(branch_input_channels, branch_channels[1], branch_layers[1], block_func, ks_2,
                               resolutions=resolution, act_func=act_func, norm_func=norm_func)
        branch_input_channels = self.MRB_Stage_2.branch_out_channels
        f_c = f_c + branch_input_channels[0]

        # STAGE 3
        self.MRB_Stage_3 = MRB(branch_input_channels, branch_channels[2], branch_layers[2], block_func, ks_3,
                               resolutions=resolution, act_func=act_func, norm_func=norm_func)

        branch_input_channels = self.MRB_Stage_3.branch_out_channels
        f_c = f_c + branch_input_channels[0]

        # STAGE 4
        self.MRB_Stage_4 = MRB(branch_input_channels, branch_channels[3], branch_layers[3], block_func, ks_4,
                               resolutions=resolution, act_func=act_func, norm_func=norm_func)

        branch_input_channels = self.MRB_Stage_4.branch_out_channels
        f_c = f_c + branch_input_channels[0]

        # the output head, namely HFCM. 
        msc_unit = []
        downsample = nn.Sequential(
            nn.Conv2d(f_c, f_c // 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_c // 3)
        )
        msc_unit.append(ResBasicBlock(f_c, f_c // 3, ksize=3, stride=1, downsample=downsample, act_func=act_func,
                                      norm_func=norm_func))

        for idx in range(head_block_nums - 1):
            msc_unit.append(ResBasicBlock(f_c // 3, f_c // 3, ksize=3, stride=1, downsample=None, act_func=act_func,
                                          norm_func=norm_func))

        self.MSFCAF = nn.Sequential(*msc_unit)

        self.head = nn.Sequential(
            nn.Conv2d(f_c // 3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            act_func(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True), 
        )
        
        self.relu_layer = nn.ReLU()

    def forward(self, x, add_dt=False, add_wl=False, is_exp=False):
        r'''
        x -> tensor(B, C, H, W): the sampled patches by MCHR, containing 5 patches in our setting, and here the C is 70 (5 * 14, while the 14 is composed of 12 channels of L2A, DtCM and WLM)
        add_dt -> bool: whether add DtCM in the input.
        add_wl -> bool: whether add WLM in the input. 
        is_exp -> bool: if true, the output is b = e^y, else b = relu(y). 
        '''
        interv = x.shape[1] // 5
        r10 = x[:, :interv, :, :]
        r30 = x[:, interv: 2 * interv, :, :]
        r90 = x[:, 2 * interv:3 * interv, :, :]
        r270 = x[:, 3 * interv: 4 * interv, :, :]
        r810 = x[:, 4 * interv: 5 * interv, :, :]

        idxs = [idx for idx in range(interv - 2)]

        if add_dt and add_wl:
            idxs.append(interv - 2)
            idxs.append(interv - 1)

        elif not add_dt and add_wl:
            idxs.append(interv - 1)

        elif add_dt and not add_wl:
            idxs.append(interv - 2)
        else:
            idxs = idxs

        r10 = r10[:, idxs, :, :]
        r30 = r30[:, idxs, :, :]
        r90 = r90[:, idxs, :, :]
        r270 = r270[:, idxs, :, :]
        r810 = r810[:, idxs, :, :]

        # 依次通过4个stage, 并再每个stage后更新除了RES-810外的分支
        # stage 1
        f1_r10, f1_r30, f1_r90, f1_r270, f1_r810 = self.MRB_Stage_1(r10, r30, r90, r270, r810)
        # stage 2
        f2_r10, f2_r30, f2_r90, f2_r270, f2_r810 = self.MRB_Stage_2(f1_r10, f1_r30, f1_r90, f1_r270, f1_r810)
        # stage 3
        f3_r10, f3_r30, f3_r90, f3_r270, f3_r810 = self.MRB_Stage_3(f2_r10, f2_r30, f2_r90, f2_r270, f2_r810)
        # stage 4
        f4_r10, f4_r30, f4_r90, f4_r270, f4_r810 = self.MRB_Stage_4(f3_r10, f3_r30, f3_r90, f3_r270, f3_r810)

        # 由r10分支输出
        fr10_cat = torch.cat([f1_r10, f2_r10, f3_r10, f4_r10], dim=1)

        pred = self.MSFCAF(fr10_cat)

        pred = self.head(pred)

        if is_exp:
            pred = torch.exp(pred)
        else:
            pred = self.relu_layer(pred)

        return pred
