r"""
The purpose of this data_reader is to read in tif data and then perform basic pre-processing like water-land map (WLM)
and the conversion from DEM to DBM.
"""
import math
import cv2
from osgeo import gdal
import numpy as np
import os
from dataloader.basic_modules import rhos_d_preprocess, rhos_wd_preprocess, invalid_pixels_fixed, reproject_aligned
# currently, the function S2R10_MCHR is not stable since we rewrite it using C++ and complied with boostpython, 
# we are working on this issue now and we will release this source code later.
import S2R10_MCHR
from joblib import dump, load
import shutil

configuration = {'thres': 0.11, 'n': 500, 'fill': True, 'bathy_valid_ratio': 0}

# os.environ['PROJ_LIB'] = r'D:\Anaconda\envs\acolite\Lib\site-packages\pyproj\proj_dir\share\proj'


# to save time, sometimes we process the data previously and saved them as '_RMDB_MAXE_XXXX.tif'
def read_rmdb(path, msc_num=12):
    r"""
    :param path: the path for the RMDB file -> string
    :param msc_num: the channels of the rhos data -> int
    :return:
    """
    file = gdal.Open(path)
    r_size, c_size = file.RasterYSize, file.RasterXSize
    rhos = np.ones([r_size, c_size, msc_num], dtype=np.float32) * -111111
    # read the rhos
    for idx in range(msc_num):
        rhos[:, :, idx] = file.GetRasterBand(idx + 1).ReadAsArray()

    # read the wlm, dt and elev
    wlm = file.GetRasterBand(msc_num + 1).ReadAsArray()
    dt = file.GetRasterBand(msc_num + 2).ReadAsArray()
    dbm = file.GetRasterBand(msc_num + 3).ReadAsArray()
    extent = file.GetGeoTransform()
    proj = file.GetProjection()

    return rhos, wlm, dt, dbm, extent, proj


def read_rmd(path, msc_num=12):
    r"""
    :param path: the path for the RMD file -> string
    :param msc_num: the channels of the rhos data -> int
    :return:
    """
    file = gdal.Open(path)
    r_size, c_size = file.RasterYSize, file.RasterXSize
    rhos = np.ones([r_size, c_size, msc_num], dtype=np.float32) * -111111
    # read the rhos
    for idx in range(msc_num):
        rhos[:, :, idx] = file.GetRasterBand(idx + 1).ReadAsArray()

    # read the wlm, dt and elev
    wlm = file.GetRasterBand(msc_num + 1).ReadAsArray()
    dt = file.GetRasterBand(msc_num + 2).ReadAsArray()
    extent = file.GetGeoTransform()
    proj = file.GetProjection()

    return rhos, wlm, dt, extent, proj


#@jit(forceobj=True)
def padding_function(rhos, mask, rotate_radius, max_scale_factor, interpolated_v=None):
    k = -111112  # here, the k is different for ndv (-111111)
    padded_rhos = np.ones([rhos.shape[0] + 2 * rotate_radius * max_scale_factor,
                           rhos.shape[1] + 2 * rotate_radius * max_scale_factor, rhos.shape[-1] - 2]) * k
    N = 100000
    B8 = rhos[:, :, 7]
    wlm = rhos[:, :, 12]
    selected_rhos = B8[B8 > 0]
    array_rhos = np.sort(selected_rhos)
    r = array_rhos[N - 1]
    candidate_coords = (B8 <= r) * (B8 > 0)  # get the coord mask that corresponding to [0, selected_rhos)
    candidate_coords = candidate_coords * (1 - mask).astype(bool)

    # here, rhos contains num_c + 2 channels, here only
    for idx in range(rhos.shape[-1] - 2):
        b = rhos[:, :, idx]
        if interpolated_v is not None:
            padded_v = interpolated_v[idx]
        else:
            # here, we using the mimimum of the bottomest 100000 pixels as the constant value to fill those invalid pixels that not covered by mask.
            padded_v = np.min(b[candidate_coords])
        # padding the rhos tensor
        padded = np.pad(b, ((rotate_radius * max_scale_factor, rotate_radius * max_scale_factor),
                        (rotate_radius * max_scale_factor, rotate_radius * max_scale_factor)), 'constant',
                        constant_values=k)
        regions_coords = padded == k
        padded[regions_coords] = padded_v
        padded_rhos[:, :, idx] = padded

    padded_wlm = np.pad(wlm, ((rotate_radius * max_scale_factor, rotate_radius * max_scale_factor),
                              (rotate_radius * max_scale_factor, rotate_radius * max_scale_factor)), 'constant',
                        constant_values=0)  # 0 for water, here we assume all padded area is water regions

    tmp = (1 - padded_wlm).astype(np.uint8)
    padded_dt = cv2.distanceTransform(tmp, distanceType=cv2.DIST_L2, maskSize=5)
    update_rhos = np.concatenate([padded_rhos, padded_wlm[:, :, None],
                                  padded_dt[:, :, None]], axis=-1)
    return update_rhos


#@jit(forceobj=True)
def padding_function_original(rhos, mask, rotate_radius, max_scale_factor, interpolated_v=None):
    k = -111112  # here, the k is different for ndv (-111111)
    padded_rhos = np.ones([rhos.shape[0] + 2 * rotate_radius * max_scale_factor,
                           rhos.shape[1] + 2 * rotate_radius * max_scale_factor, rhos.shape[-1] - 2]) * k
    wlm = rhos[:, :, 12]
    N = 100000
    B8 = rhos[:, :, 7]
    selected_rhos = B8[B8 > 0]
    selected_rhos = np.sort(selected_rhos)[N - 1]
    coord = (B8 <= selected_rhos) * (B8 > 0)

    # here, rhos contains num_c + 2 channels, here only
    for idx in range(rhos.shape[-1] - 2):
        b = rhos[:, :, idx]
        added_rhos = np.mean(b[coord])
        padded = np.pad(b, ((rotate_radius * max_scale_factor, rotate_radius * max_scale_factor),
                        (rotate_radius * max_scale_factor, rotate_radius * max_scale_factor)), 'constant',
                        constant_values=-111111)
        padded[padded == -111111] = added_rhos
        padded_rhos[:, :, idx] = padded

    padded_wlm = np.pad(wlm, ((rotate_radius * max_scale_factor, rotate_radius * max_scale_factor),
                              (rotate_radius * max_scale_factor, rotate_radius * max_scale_factor)), 'constant',
                        constant_values=0)  # 0 for water, here we assume all padded area is water regions

    tmp = (1 - padded_wlm).astype(np.uint8)
    padded_dt = cv2.distanceTransform(tmp, distanceType=cv2.DIST_L2, maskSize=5)
    update_rhos = np.concatenate([padded_rhos, padded_wlm[:, :, None],
                                  padded_dt[:, :, None]], axis=-1)
    return update_rhos



def read_list(path):
    files_list = os.listdir(path)

    for idx in range(len(files_list)):
        p = path + '/' + files_list[idx]
        p = p.replace('//', '/').replace('\\', '/')
        files_list[idx] = p

    return files_list


class DataReadIn():
    def __init__(self, path_list, process_type, c_num=12, tmp_dict=None, is_dbm=True, max_bathy=25, stride=10,
                 is_random_shift=True, random_shift_scale=4, patch_size=15, mb_res=np.array([10, 30, 90, 270, 810]),
                 basic_res=10, data_dict=None):
        # first, we read all data and create a tmp dict to save thess data, because it will waste too much ram
        file_num = len(path_list)
        # here, the rhos, wlm and dt is concatenated, and dbm is stored in bathy_list, and some other information like
        # extent, proj and filenames are stored in info_list
        radius = math.ceil(patch_size // 2 * math.sqrt(2))  # the reason for the sqrt(2) is for further rotation
        #
        while (2 * radius + 1) % 3 != 0:
            radius += 1
        #
        print('The expanded radius is: %d.' % radius)
        self.radius = radius
        self.crop_radius = patch_size // 2
        max_scale = np.max(mb_res) // basic_res
        self.padding_v = max_scale * self.radius
        self.file_path = []
        if tmp_dict is None:
            self.tmp_dict = os.getcwd().replace('\\', '/')
        else:
            self.tmp_dict = tmp_dict
            if not os.path.exists(self.tmp_dict):
                os.makedirs(self.tmp_dict, exist_ok=True)
        if data_dict is not None and os.path.exists(data_dict):
            # if the data have processed and stored in the dict, just read it and skip all others steps
            file_path = read_list(data_dict)
        else:
            file_path = []

        if len(file_path) > 0:
            self.file_path = file_path
        else:
            # we need read the data from the start
            # there are file_num files need to be read in
            for idx in range(file_num):
                element = path_list[idx]  # element should be a dict
                print('The file is processing: ' + element['rhos'] + '] -> [%d / %d]' % (idx + 1, file_num))
                # if it is read from _RMDB_ files
                if process_type == 'RMDB':
                    active_region = reproject_aligned(element['rhos'], element['active_region'])
                    active_region = active_region == -111111
                    rhos, wlm, dt, dbm, extent, proj = read_rmdb(element['rhos'], c_num)
                    # masking the dbm based on the active_region
                    dbm[active_region] = -111111
                    mask = reproject_aligned(element['rhos'], element['mask'])
                    mask = mask != -111111
                    rhos, interpolated_v = invalid_pixels_fixed(rhos, mask, ndv=-111111.)  # get the filled rhos data
                    data = np.concatenate([rhos, wlm[:, :, None], dt[:, :, None]], axis=-1)  # added with other 2 channel
                    data = data.astype(np.float32)
                    data = padding_function_original(data, mask, self.radius, max_scale_factor=max_scale, interpolated_v=None)
                    filename = element['rhos'].replace('//', '/').replace('\\', '/')
                    filename = filename.split('/')[-1]  # get the file name and store it
                    info_ = {'extent': extent, 'proj': proj, 'filename': filename, 'R': dbm.shape[0], 'C': dbm.shape[1]}
                    # data = np.ascontiguousarray(data)  # necessary step for further MCHR
                    # save data to tmp dict
                    filename = filename.split('.tif')[0]
                    tmp_path = self.tmp_dict + '/' + filename + '.pkl'
                    dump({'data': data, 'bathy': dbm, 'info': info_}, tmp_path, compress=3)
                    self.file_path.append(tmp_path)

                elif process_type == 'RMD':  # RMD and RAW_WD do not have active_region and dem
                    # active_region = reproject_aligned(element['rhos'], element['active_region'])
                    rhos, wlm, dt, extent, proj = read_rmd(element, c_num)
                    mask = reproject_aligned(element['rhos'], element['mask'])
                    rhos, interpolated_v = invalid_pixels_fixed(rhos, mask, ndv=-111111.)  # get the filled rhos data
                    data = np.concatenate([rhos, wlm[:, :, None], dt[:, :, None]], axis=-1)  # added with other 2 channel
                    data = data.astype(np.float32)
                    data = padding_function_original(data, self.radius, max_scale_factor=max_scale, interpolated_v=interpolated_v)
                    filename = element.split('/').split('\\').split('//').split('.')[-2]  # get the file name and store it
                    info_ = {'extent': extent, 'proj': proj, 'filename': filename, 'R': rhos.shape[0], 'C': rhos.shape[1]}
                    data = np.ascontiguousarray(data)  # necessary step for further MCHR
                    # save data to tmp dict
                    filename = filename.split('.tif')[0]
                    tmp_path = self.tmp_dict + '/' + filename + '.pkl'
                    dump({'data': data, 'bathy': None, 'info': info_}, tmp_path, compress=3)
                    self.file_path.append(tmp_path)

                elif process_type == 'RAW_D':
                    # if it is RAW_D, then the element composed of three path, that is, the path for rhos, dem and mask,
                    # and we need some configuration for the process, such as thres and n.
                    rhos_path = element['rhos']
                    dem_path = element['dem']
                    mask_path = element['mask']
                    active_region = reproject_aligned(rhos_path, element['active_region'])
                    active_region = active_region == -111111
                    filename = rhos_path.split('/').split('\\').split('//').split('.')[-2]  # get the file name and store it
                    # it has been processed and invalid pixels are filled.
                    rhos, wlm, dt, dbm, extent, proj = rhos_d_preprocess(raster_path=rhos_path, dem_path=dem_path,
                                                                         mask_path=mask_path, thres=configuration['thres'],
                                                                         n=configuration['n'],
                                                                         nan_filter=configuration['fill'],
                                                                         is_output=False)
                    dbm = active_region * dbm
                    data = np.concatenate([rhos, wlm[:, :, None], dt[:, :, None]], axis=-1)  # cat with wlm (M) and dtcm (D)
                    data = data.astype(np.float32)
                    data = padding_function_original(data, self.radius, max_scale_factor=max_scale)
                    data = np.ascontiguousarray(data)  # necessary step for further MCHR
                    info_ = {'extent': extent, 'proj': proj, 'filename': filename, 'R': dbm.shape[0], 'C': dbm.shape[1]}
                    # save data to tmp dict
                    filename = filename.split('.tif')[0]
                    tmp_path = self.tmp_dict + '/' + filename + '.pkl'
                    dump({'data': data, 'bathy': dbm, 'info': info_}, tmp_path, compress=3)
                    self.file_path.append(tmp_path)

                elif process_type == 'RAW_WD':
                    # if it is RAW_WD, then the element composed of two path, including the path for rhos and mask
                    rhos_path = element['rhos']
                    mask_path = element['mask']
                    filename = rhos_path.split('/').split('\\').split('//').split('.')[-2]  # get the file name and store it
                    rhos, wlm, dt, extent, proj = rhos_wd_preprocess(raster_path=rhos_path, mask_path=mask_path,
                                                                     thres=configuration['thres'], is_output=False)

                    data = np.concatenate([rhos, wlm[:, :, None], dt[:, :, None]], axis=-1)
                    data = data.astype(np.float32)
                    data = padding_function_original(data, self.radius, max_scale_factor=max_scale)
                    data = np.ascontiguousarray(data)  # necessary step for further MCHR
                    info_ = {'extent': extent, 'proj': proj, 'filename': filename, 'R': rhos.shape[0], 'C': rhos.shape[1]}
                    # save data to tmp dict
                    filename = filename.split('.tif')[0]
                    tmp_path = self.tmp_dict + '/' + filename + '.pkl'
                    dump({'data': data, 'bathy': None, 'info': info_}, tmp_path, protocol=3)
                    self.file_path.append(tmp_path)

                else:
                    print('The process_type is un-defined, check it.')
                    exit(-1)

        self.is_dbm = is_dbm
        self.is_random_shift = is_random_shift
        self.random_shift_scale = random_shift_scale
        self.max_bathy = max_bathy
        self.process_type = process_type
        self.stride = stride
        self.mb_res = np.ascontiguousarray(mb_res.astype(np.int32))  # make sure it is int32
        self.coords_list = None

        self.data = None
        self.bathy = None
        self.info_ = None

        # perform the MCHR at first
        self.sampler()

    def sampler(self):
        print("\n ############ Updating the dataset ############ ")
        if self.is_dbm:
            # if it is training, it must have dem file, some process_type must be RAW_D or RMDB
            # set the info_, data and bathy as empty list
            self.info_ = []
            self.data = []
            self.bathy = []
            self.info_list = []
            assert (self.process_type == 'RMDB' or self.process_type == 'RAW_D'), "It must be RAW_D or RMDB in training"
            for idx in range(len(self.file_path)):
                #
                print('The processed file is: ' + self.file_path[idx] + ' [%d / %d]' % (idx + 1, len(self.file_path)))
                f = load(self.file_path[idx])
                bathy_map = f['bathy']
                rhos = f['data']
                self.info_list.append(f['info'])
                del f
                bathy_map[bathy_map > self.max_bathy] = -111111  # set the maximum bathymetry data
                h, w = rhos.shape[0], rhos.shape[1]
                # the coordinates used for sample, here it should be the idx in padded rhos
                r_coords = []
                c_coords = []
                offset = self.padding_v + self.radius + self.random_shift_scale
                tmp_list = []
                for r_idx in range(offset, h - offset, self.stride):
                    for c_idx in range(offset, w - offset, self.stride):
                        R = r_idx
                        C = c_idx

                        if self.is_random_shift:
                            # shifted coordinates
                            R = R + np.random.randint(-self.random_shift_scale, self.random_shift_scale + 1, 1)[0]
                            C = C + np.random.randint(-self.random_shift_scale, self.random_shift_scale + 1, 1)[0]

                        # get the bathymetric patch
                        r_for_bathy, c_for_bathy = R - self.padding_v, C - self.padding_v
                        # to ensure not exceed boundary, if r and c smaller than 0, setting them as 0
                        # get the cropped patch of bathymetry
                        crop_bathy_patch = bathy_map[r_for_bathy - self.crop_radius: r_for_bathy + self.crop_radius + 1,
                                                     c_for_bathy - self.crop_radius: c_for_bathy + self.crop_radius + 1]

                        valid_ratio = np.sum(crop_bathy_patch != -111111.) / ((2 * self.crop_radius + 1) ** 2)

                        tmp_list.append([r_for_bathy, c_for_bathy, valid_ratio])

                        if valid_ratio > configuration['bathy_valid_ratio']:
                            r_coords.append(R)
                            c_coords.append(C)

                # for MCHR
                r_coords = np.array(r_coords)
                c_coords = np.array(c_coords)
                coords = np.concatenate([r_coords[:, None], c_coords[:, None]], axis=-1)
                # obtain the resampled data
                rhos = np.ascontiguousarray(rhos)
                coords = np.ascontiguousarray(coords)
                coords = coords.astype(np.int32)
                rhos = rhos.astype(np.float32)

                mchr_data_list, coords_list = S2R10_MCHR.S2R10_MCHR(rhos, coords, self.mb_res, self.radius, 0.75)
                
                # imgs = patch_imgs_creator(mchr_data_list)
                # imgs_write(imgs, 'G:/PatchVisualize')

                # correct the coordinates and change it in un-padded datum
                for iidx in range(len(coords_list)):
                    mchr_r, mchr_c = coords_list[iidx][0], coords_list[iidx][1]
                    self.data.append(mchr_data_list[iidx])
                    r_for_bathy, c_for_bathy = mchr_r - self.padding_v, mchr_c - self.padding_v
                    bathy_patch = bathy_map[r_for_bathy - self.radius: r_for_bathy + self.radius + 1,
                                            c_for_bathy - self.radius: c_for_bathy + self.radius + 1]
                    self.bathy.append(bathy_patch)
                    self.info_.append({'r': r_for_bathy, 'c': c_for_bathy, 's_idx': idx})  # idx is for the scene
                    # obtain the sampled data, bathy and info_

                del rhos, coords, bathy_map
        return print('The MCHR have been finished.')
    
    def _clear(self):
        self.data = None
        self.bathy = None
        self.info_ = None
        
        return 
    
    def delete_all(self):
        shutil.rmtree(self.tmp_dict)

        return print('The data are all removed from tmp_dict.')
