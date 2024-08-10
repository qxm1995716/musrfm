import os
import numpy as np
from utils.raster_generator import raster_output


def bms_embedding(embedded_maps, info_, predicts, wlms, radius=None, ndv=-111111):
    # here, embedded_maps is a list of multiple bathymetry maps, corresponding to files come from different region or
    # period.
    h, w = predicts.shape[-2], predicts.shape[-1]
    b = predicts.shape[0]
    predicts = predicts.clone().detach().cpu()
    predicts = np.array(predicts)
    wlms = np.array(wlms.tolist())
    wlms = np.array(wlms)

    if radius is None:
        radius = h // 2

    if len(predicts.shape) == 4:
        predicts = predicts.squeeze(1)

    for idx in range(b):
        scene_idx = info_['s_idx'][idx]
        # here, the r and c are coords in the base of un-padded rhos.
        r = info_['r'][idx]
        c = info_['c'][idx]
        wlm = wlms[idx, :, :] == 1
        p = predicts[idx, :, :]
        p[wlm] = ndv  # pixels of land are all set to be zero
        embedded_maps[scene_idx][r - radius: r + radius + 1, c - radius: c + radius + 1] = p

    return embedded_maps


def bms_write(info_list, bms, save_dict):
    n = len(info_list)
    if not os.path.exists(save_dict):
        os.makedirs(save_dict, exist_ok=True)
    #
    for idx in range(n):
        fname = info_list[idx]['filename']
        extent = info_list[idx]['extent']
        proj = info_list[idx]['proj']
        save_path = save_dict + '/' + fname

        if '.tif' not in save_path:
            save_path = save_path + '.tif'

        raster_output(save_path, [bms[idx]], extent=extent, proj=proj, ndv=-111111)

    return print('The raster data has output to the specific dict.')


def bms_create(info_list, ndv=-111111):
    n = len(info_list)
    bms = []
    for idx in range(n):
        RSize, CSize = info_list[idx]['R'], info_list[idx]['C']
        bm = np.ones([RSize, CSize], dtype=np.float32) * ndv
        bms.append(bm)

    return bms
