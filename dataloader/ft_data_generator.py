import os
from osgeo import gdal
import numpy as np
from utils.raster_generator import raster_output
import random
from copy import deepcopy


def random_ft_extract(gt_raster, ft_nums):
    valid_gt_coords = np.argwhere(gt_raster != -111111)
    valid_nums = valid_gt_coords.shape[0]
    indices_list = [idx for idx in range(valid_nums)]
    #
    selected_indices = random.sample(indices_list, ft_nums)
    coords_for_ft_samples = valid_gt_coords[selected_indices]
    #
    mask = np.zeros_like(gt_raster)
    mask[coords_for_ft_samples[:, 0], coords_for_ft_samples[:, 1]] = 1
    #
    ft_gt_raster = deepcopy(gt_raster)
    ft_gt_raster[mask != 1] = -111111
    gt_raster[mask == 1] = -111111

    return gt_raster, ft_gt_raster


def generator_data(processed_raster_path, ft_nums, out_path, max_bathy=25):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #
    raster = gdal.Open(processed_raster_path)
    fname = processed_raster_path.split('/')[-1]
    fname = fname.split('.tif')[0]
    rhos = []
    extent = raster.GetGeoTransform()
    proj = raster.GetProjection()
    raster_count = raster.RasterCount
    for idx in range(raster_count - 1):
        R = raster.GetRasterBand(idx + 1).ReadAsArray()
        R = R[:, :, None]
        rhos.append(R)

    rhos = np.concatenate(rhos, axis=-1)

    gt = raster.GetRasterBand(raster_count).ReadAsArray()
    gt[gt > max_bathy] = -111111
    inf_gt, ft_gt = random_ft_extract(gt, ft_nums)

    inf_data = np.concatenate([rhos, inf_gt[:, :, None]], axis=-1)
    ft_data = np.concatenate([rhos, ft_gt[:, :, None]], axis=-1)

    inf_name = fname + '_INFER.tif'
    ft_name = fname + '_FT.tif'
    raster_output(out_path + '/' + inf_name, [inf_data], extent=extent, proj=proj)
    raster_output(out_path + '/' + ft_name, [ft_data], extent=extent, proj=proj)

    return


if __name__ == '__main__':
    kauai_path = r'G:/Data Files/NCMP/PREPROCESSED_DATA_NT/DEMV2_PREPROCESSED_DATA_T011N2500/NCMP_20180101_20220101_AB_MED_RMDB_MAXE_1.9965.tif'
    generator_data(kauai_path, ft_nums=10000, out_path='../data/KAUAI')