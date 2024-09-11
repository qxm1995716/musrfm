import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import logger
from dataloader.data_reader import DataReadIn
from dataloader.Loader import dataset_creator
from data.creat_list import rdm_filelist, rm_filelist
from timm.scheduler.cosine_lr import CosineLRScheduler
from models.MRFM import MRNet
from utils.tools import bms_create, bms_embedding, bms_write
from utils.metrics import non_overlap_mse
import argparse
import math
from utils.loss_func import MaskedMAE_SSMI, MaskedMSE_SSMI, MaskedMSE, MaskedMAE
from models.BasicUnits import ResBasicBlock
import time


def args():
    paras = argparse.ArgumentParser('MRFS_Adabins')
    # basic parameters
    r = 7
    patchsize = 2 * r + 1
    stride = 11
    if stride % 2 != 0:
        stride = stride - 1

    test_stride = 11
    if test_stride % 2 == 0:
        test_stride = test_stride - 1
    embedding_radius = 5

    mb_res = np.array([10, 30, 90, 270, 810])
    paras.add_argument('--radius', type=int, default=r, help='The radius used in the MCHR.')
    paras.add_argument('--patch_size', type=int, default=patchsize, help='The H and W of patch of MCHR.')
    paras.add_argument('--c_num', type=int, default=12, help='The channels of multi-spectral imagery.')
    paras.add_argument('--max_bathy', type=float, default=25., help='The maximum bathymetry value used in SDB.')
    paras.add_argument('--random_shift_scale', type=int, default=2, help='The shift scale used in MCHR.')
    paras.add_argument('--stride', type=int, default=stride, help='The stride for shift window.')
    paras.add_argument('--mb_res', type=np.ndarray, default=mb_res, help='The resolution of different branches.')
    paras.add_argument('--test_stride', type=int, default=test_stride)
    paras.add_argument('--embedding_radius', type=int, default=embedding_radius)
    paras.add_argument('--mchr_update_interv', type=int, default=1, help='The interval for update samples')
    paras.add_argument('--mrf_level_nums', type=int, default=len(mb_res))

    # training parameters
    paras.add_argument('--batch_size', type=int, default=256, help='The Batch Size used in training.')
    paras.add_argument('--epoch', type=int, default=15, help='The number of epoch for training.')
    paras.add_argument('--lr_scheduler_epoch', type=int, default=9, help='The number of epoch for lr scheduler.')
    paras.add_argument('--cycle_limit', type=float, default=1)
    paras.add_argument('--cycle_mul', type=float, default=1.0)
    paras.add_argument('--cycle_decay', type=float, default=0.5)
    paras.add_argument('--optimizer', type=str, default='AdamW', help='The optimizer used in the experiment.')
    paras.add_argument('--scheduler', type=str, default='cosine', help='The scheduler for training model.')
    paras.add_argument('--lr', type=float, default=2.5e-4, help='The initial learning rate.')
    paras.add_argument('--warmup_lr', type=float, default=1e-3, help='The learning rate for warm up.')
    paras.add_argument('--warmup_epoch', type=int, default=0, help='The epoch for the warm up period.')
    paras.add_argument('--min_lr', type=float, default=1e-7, help='The minimum learning rate for lr scheduler.')
    paras.add_argument('--update_limit', type=int, default=160,
                       help='The samples update only performed before this epoch, and after this the dataset is fixed.')
    paras.add_argument('--update_stride', type=int, default=stride // 2)

    # model's parameter
    paras.add_argument('--add_wlm', type=bool, default=False, help='Whether add WLM in the input of model.')
    paras.add_argument('--add_dtcm', type=bool, default=False, help='Whether add DtCM in the input of model.')
    paras.add_argument('--branch_channels', type=list, default=[[64, 64, 64, 48, 48], [128, 128, 128, 96, 96], [192, 192, 128, 128, 128],
                                                                [384, 384, 192, 192, 192]])
    paras.add_argument('--branch_layers', type=list, default=[[2, 2, 2, 2, 2], [3, 3, 3, 2, 2], [3, 3, 2, 2, 2], [3, 3, 2, 2, 2]])
    paras.add_argument('--kernel_size', type=list, default=[3, 3, 5, 3])
    paras.add_argument('--block_func', default=ResBasicBlock)
    paras.add_argument('--act_func', default=nn.LeakyReLU)
    paras.add_argument('--norm_func', default=nn.BatchNorm2d)
    paras.add_argument('--head_block_nums', default=3)
    paras.add_argument('--max_depth', default=25)

    return paras.parse_args()


def mbr_creator(inputs, num_branches, bands_idxs, add_wlm, add_dtcm):
    num_c = inputs.shape[1]
    num_c_per = num_c // num_branches
    mbr_inputs = []
    # wlm = np.array(inputs[:, 12, :, :].tolist())
    if add_wlm:
        bands_idxs.append(num_c_per - 2)  # 倒数第二个维度是WLM
    if add_dtcm:
        bands_idxs.append(num_c_per - 1)  # 最后一个维度是DtCM
    # the input of different resolutions
    for idx in range(num_branches):
        selected_res = inputs[:, idx * num_c_per: (idx + 1) * num_c_per, :, :]
        mbr_inputs.append(selected_res[:, bands_idxs, :, :])

    return mbr_inputs


def train_on_batch(model, data, gt, loss_func, optimizer, radius, Log, iter_idx, iter_nums, epoch_idx, epoch_nums):
    data = data.cuda()
    gt = gt.cuda()
    optimizer.zero_grad()
    predictions = model(data)
    loss = loss_func(predictions, gt)
    loss.backward()
    optimizer.step()
    mse = non_overlap_mse(predictions, gt, radius=radius)
    mse = float(mse[0].item() / mse[1].item())
    Log.log_string('[TRAINING epoch %d / %d] --- The Non-Overlap MSE get on iteration %d [total: %d] is: %.6f  [MSE: %.6f / RMSE: %.6f]'
                   % (epoch_idx + 1, epoch_nums, iter_idx + 1, iter_nums, float(loss.item()), mse, math.sqrt(mse)))

    return float(loss.item())


def test_on_batch(model, data, gt, geo_infos, test_raster, gt_raster, radius, Log, wlm):
    data = data.cuda()
    gt = gt.cuda()
    predictions = model(data)
    if radius > predictions.shape[-1] // 2:
        radius = predictions.shape[-1] // 2
    loss = non_overlap_mse(predictions, gt, radius=radius)
    test_raster = bms_embedding(test_raster, geo_infos, predictions, wlm, radius=radius)
    gt_raster = bms_embedding(gt_raster, geo_infos, gt, wlm, radius=radius)
    Log.log_string('[TESTING] --- The None-Overlap MSE get on this batch is: %.6f' % float(loss[0].item() / loss[1].item()))

    return test_raster, gt_raster


def main_train(args, Log):
    in_channels = args.c_num
    if args.add_wlm:
        in_channels += 1

    if args.add_dtcm:
        in_channels += 1
    
    model = MRNet(
        in_channels=in_channels,
        branch_channels=args.branch_channels,
        branch_layers=args.branch_layers,
        kernel_size=args.kernel_size,
        block_func=args.block_func,
        act_func=args.act_func,
        head_block_nums=args.head_block_nums
    )
    
    model.cuda()

    optimizer = None
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # all other's parameters are default.
    else:
        print('Wrong setting for optimizer.')
        exit(-1)

    schedular = CosineLRScheduler(optimizer, t_initial=args.lr_scheduler_epoch, cycle_limit=args.cycle_limit,
                                  lr_min=args.min_lr, warmup_lr_init=args.warmup_lr, warmup_t=args.warmup_epoch,
                                  cycle_mul=args.cycle_mul, cycle_decay=args.cycle_decay, t_in_epochs=True)

    # loading train and test datasets.
    patch_size = 2 * args.radius + 1
    # why we set the expanded_radius? because when we perform random roatation for data enhancement, if the size of 
    # patch is patch_size, then there will have some invalid pixels in the roatated patches. So we first expand the
    # range of shape and then perform central crop function to get the final input.
    expanded_radius = math.ceil(patch_size // 2 * math.sqrt(2))
    while (2 * expanded_radius + 1) % 3 != 0:
        expanded_radius += 1
    #
    train_path = './data/train.txt'
    train_files = rm_filelist(train_path)
    test_path = './data/test.txt'
    test_files = rm_filelist(test_path)

    tmp_train_path = './PFOTMP_TRAIN_EPR' + str(expanded_radius) + '_MR' + str(args.mb_res[-1])  # PFO for padding_function_original
    tmp_test_path = './PFOTMP_TEST_EPR' + str(expanded_radius) + '_MR' + str(args.mb_res[-1])

    Log.log_string('\n[DATALOADER] --- Loading training dataset from files. ---')
    trains_container = DataReadIn(train_files, process_type='RMDB', c_num=args.c_num, is_dbm=True, is_random_shift=False,
                                  random_shift_scale=args.random_shift_scale, tmp_dict=tmp_train_path,
                                  stride=args.stride, max_bathy=args.max_depth, patch_size=args.patch_size,
                                  mb_res=args.mb_res, basic_res=10, data_dict=tmp_train_path)

    train_datasets = dataset_creator(trains_container, batch_size=args.batch_size, radius=args.radius,
                                     is_training=True, is_distributed=False)
    train_dataloader, train_geo_infos = train_datasets.dataloader, train_datasets.geo_info

    Log.log_string('\n[DATALOADER] --- Loading testing dataset from files. ---')
    tests_container = DataReadIn(test_files, process_type='RMDB', c_num=args.c_num, is_dbm=True, is_random_shift=False,
                                 random_shift_scale=args.random_shift_scale, tmp_dict=tmp_test_path,
                                 stride=args.test_stride, max_bathy=args.max_depth, patch_size=args.patch_size,
                                 mb_res=args.mb_res, basic_res=10, data_dict=tmp_test_path)

    test_datasets = dataset_creator(tests_container, batch_size=args.batch_size, radius=args.radius,
                                    is_training=False, is_distributed=False)
    test_dataloader, test_geo_infos = test_datasets.dataloader, test_datasets.geo_info

    # loss function setting
    loss_func = MaskedMSE()
    # creat the embedding raster
    test_raster_embedded = bms_create(tests_container.info_list, ndv=-111111)
    test_gt_embedded = bms_create(tests_container.info_list, ndv=-111111)
    # training process
    loss_from_epochs = []
    for idx in range(args.epoch):
        model.train()
        loss_sum = 0
        start_time = time.time()
        for iter_idx, (data, gt, patch_geo_infos) in enumerate(train_dataloader):
            loss_value = train_on_batch(model, data, gt, loss_func, optimizer, args.radius, Log, iter_idx,
                                        len(train_dataloader), idx, args.epoch)
            loss_sum += loss_value
        schedular.step(idx + 1)
        loss_mean = loss_sum / len(train_dataloader)
        loss_from_epochs.append(loss_mean)
        used_time = time.time() - start_time
        Log.log_string('[TRAINING] --- The mean loss got on this epoch is: %.6f, used time is: %.6f.' % (loss_mean, used_time))

    # perform testing
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        Log.log_string('\n ##### The evaluation of model on testing dataset ##### ')
        for iter_idx, (data, gt, patch_geo_infos) in enumerate(test_dataloader):
            wlm = data[:, 12, :, :]
            test_raster_embedded, test_gt_embedded = test_on_batch(model, data, gt, patch_geo_infos,
                                                                   test_raster_embedded, test_gt_embedded,
                                                                   args.embedding_radius, Log, wlm)
        used_time = time.time() - start_time
        mse_metric = 0
        num_samples = 0

        for s_idx in range(len(test_raster_embedded)):
            region_mask = test_gt_embedded[s_idx] != -111111
            mse_metric += np.sum((test_raster_embedded[s_idx][region_mask] - test_gt_embedded[s_idx][region_mask]) ** 2)
            num_samples += np.sum(region_mask)
        mse = mse_metric / num_samples
        rmse = math.sqrt(mse)
        Log.log_string('The MSE and RMSE obtained on all results is: %.6f and %.6f [used time: %.6f].' %
                       (mse, rmse, used_time))
        
    #
    bms_write(tests_container.info_list, test_raster_embedded, str(Log.tif_dict) + '/preds/')
    bms_write(tests_container.info_list, test_gt_embedded, str(Log.tif_dict) + '/gts/')

    # save the model
    state = {'model_parameters': model.state_dict()}
    save_path = str(Log.checkpoint_dict) + '/model.pth'
    torch.save(state, save_path)

    return


if __name__ == '__main__':
    args = args()
    Log = logger('MRF_ORGINIAL')
    main_train(args, Log)
