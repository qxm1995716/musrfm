import copy
from tqdm import tqdm
import numpy as np
import os
from osgeo import gdal, gdalconst
import cv2
from numba import jit
from scipy.special import erfcinv
import math
from utils.raster_generator import raster_output
from scipy.interpolate import griddata


def reproject_aligned(target_raster, project_raster):
    target = gdal.Open(target_raster)
    project_object = gdal.Open(project_raster)

    target_extent = target.GetGeoTransform()
    target_proj = target.GetProjection()
    project_object_proj = project_object.GetProjection()

    # elevation map
    tmp_raster = r'./tmp.tif'
    if os.path.exists(tmp_raster):
        os.remove(tmp_raster)
    driver = gdal.GetDriverByName('GTiff')
    handler = driver.Create(tmp_raster, target.RasterXSize, target.RasterYSize, 1, gdal.GDT_Float32)
    handler.SetGeoTransform(target_extent)
    handler.SetProjection(target_proj)

    handler.GetRasterBand(1).SetNoDataValue(-111111)
    # perform re-projection for dem, to align with the rhos
    gdal.ReprojectImage(project_object, handler, project_object_proj, target_proj, gdalconst.GRA_Bilinear)
    projected_map = handler.GetRasterBand(1).ReadAsArray()

    handler = None
    # os.remove(tmp_raster)

    return projected_map


# the function that used for interpolate nan pixels
@jit(forceobj=True)
def nan_interp(data, nan_coords, thres, ndv, offset=0, max_radius=50):
    r"""
    :param data: the rhos raster data, which is only one channel -> np::ndarray -> [h w]
    :param nan_coords: the coords list for nan pixels -> list[np::ndarray] -> [N]
    :param thres: the threshold for -> float
    :param ndv: the value of nan pixels -> float
    :param offset: the offset value is the limitation of the position of coord -> int
    :param max_radius: the radius of the selected patch -> int
    :return:
    new_data: the rhos raster that have been interpolated -> np::ndarray -> [h w]
    """
    nums = nan_coords[0].shape[0]
    new_data = copy.deepcopy(data)

    for idx in tqdm(range(nums)):
        coords_r = nan_coords[0][idx]
        coords_c = nan_coords[1][idx]
        func = True
        # the initial radius is 3
        radius = 3
        # iteration
        while func and data.shape[0] - offset > coords_r > offset and offset < coords_c < data.shape[1] - offset:
            # get the selected patch
            patch = data[coords_r - radius: coords_r + radius + 1, coords_c - radius: coords_c + radius + 1]
            # the number of pixels that is ndv or nan
            count = np.sum(patch == ndv) + np.sum(np.isnan(patch))
            # to make sure that there have enough pixels that have valid values in the patch
            if count / (2 * radius + 1) ** 2 < thres:
                func = False
                indices_r = np.zeros([2 * radius + 1, 2 * radius + 1])
                indices_c = np.zeros([2 * radius + 1, 2 * radius + 1])
                # build the matrix corresponding to row and column for further interpolation
                for iidx in range(2 * radius + 1):
                    indices_r[iidx, :] = iidx
                    indices_c[:, iidx] = iidx
                # transfer to bool matrix
                v = (1 - np.isnan(patch)).astype(bool)  # the bool matrix for non-nan
                vv = patch != ndv  # the bool matrix for non-ndv
                cv = v * vv  # concat these two matrix
                # get the x y z
                x = indices_r[cv]
                y = indices_c[cv]
                z = patch[cv]
                # get the interpolated value of the center of patch by griddata function
                target_v = griddata((x, y), z, ([radius], [radius]), 'linear')
                # set the interpolated value in rhos matrix
                new_data[coords_r, coords_c] = target_v
            else:  # if there are too much nan or ndv in current patch, then enlarge it
                if radius > max_radius:
                    break
                radius += 1

    return new_data


# MAD outlier detection algorithm
def MAD(data, coords, n):
    r"""
    :param data: the sequence of observations
    :param coords: the coordinate for these observations
    :param n: a hype-parameter set for outlier detection
    :return:
    remove_coords: the coords that need to be removed
    median: the median value of the data
    mad: the mad parameter obtained from data
    """
    median = np.median(data)
    deviations = abs(data - median)
    c = -1 / (math.sqrt(2) * erfcinv(3 / 2))
    # 使用c来求得换算MAD
    mad = c * np.median(deviations)
    remove_idx = np.where(abs(data - median) > n * mad)
    remove_coords = [coords[0][remove_idx], coords[1][remove_idx]]
    return remove_coords, median, mad


# function used to eliminate outliers in coastal line
def CORRECT_DBM(dt, mask, dem, n=3, min_depth=-5):
    r"""
    :param dt: DtCM data obtained from WLM -> np::ndarray -> [h w]
    :param mask: WLM, it should be noticed that the element 1 is land and element 0 is water -> np::ndarray -> [h w]
    :param dem: the elevations -> np::ndarray -> [h w]
    :param n: the hype-parameter for MAD algorithm -> float
    :param min_depth: the minimum elevations that included in the sequence for outlier detection -> int
    :return:
    update_dem: the dem matrix that updated -> np::ndarray -> [h w]
    mask: the updated WLM, and the element 1 is land and element 0 is water -> np::ndaarray -> [h w]
    """
    is_outlier = True

    while is_outlier:
        # costal should be coastal
        # 确定所有海岸线的坐标
        costal_coords = np.where(dt == 1)
        costal_line = dem[costal_coords[0], costal_coords[1]]
        costal_coords = [costal_coords[0][(costal_line != -111111) * (costal_line > min_depth)],
                         costal_coords[1][(costal_line != -111111) * (costal_line > min_depth)]]
        costal_line = costal_line[(costal_line != -111111) * (costal_line > min_depth)]

        # filtered_coastal_line = []
        _, median, cmad = MAD(costal_line, costal_coords, n)
        filter_coords_x = []
        filter_coords_y = []

        # here, we only consider that the elevation are larger than [median + n * cmad]
        for idx in range(len(costal_coords[0])):
            if costal_line[idx] > median + n * cmad:
                filter_coords_x.append(costal_coords[0][idx])
                filter_coords_y.append(costal_coords[1][idx])

        # if there are some outliers exist
        if len(filter_coords_x) != 0:
            remove_x = np.array(filter_coords_x)
            remove_y = np.array(filter_coords_y)
            # update the WLM
            mask[remove_x, remove_y] = 1
            # reverse the WLM to make it can be processed by further cv2.distanceTransform
            rmask = 1 - mask
            rmask = rmask.astype(np.uint8)
            dt = cv2.distanceTransform(rmask, distanceType=cv2.DIST_L2, maskSize=5)
        else:  # if there are no outliers, the iteration is finished
            is_outlier = False

    # construct the update dem
    update_dem = np.ones([dem.shape[0], dem.shape[1]]) * -111111
    update_dem[mask == 0] = dem[mask == 0]
    print('The filter program is finished.')

    return update_dem, mask


# a simple binary threshold segmentation algorithm
@jit(forceobj=True)
def Binary_Threshold(nir, v, below_idx, above_idx):
    H, W = nir.shape[0], nir.shape[1]
    mask = np.zeros([H, W])
    for r_idx in range(H):
        for c_idx in range(W):
            if nir[r_idx, c_idx] > v:
                mask[r_idx, c_idx] = above_idx
            else:
                mask[r_idx, c_idx] = below_idx

    return mask


# Segment the matrix into binary classes, one is the label that is the most and the rest are classed as the other one
@jit(forceobj=True)
def PrimElement(label, num_objects):
    r"""
    :param label: the raster that obtained by cv2.connectedComponents, it classes the matrix into different regions
                  that are not connected, with different labels -> np::ndarray -> [h w]
    :param num_objects: the number of labels that exist in label -> int
    :return:
    np_array: it's name has no specific means, which is the binary class mask obtained by the function -> np::ndarray
              -> [h w]
    """
    np_array = np.ones(label.shape)
    count_array = np.zeros([num_objects, ])
    h = label.shape[0]
    w = label.shape[1]

    for r_idx in range(h):
        for c_idx in range(w):
            element = label[r_idx, c_idx]
            # it used to record the number of each label
            count_array[int(element)] += 1

    # get the most label
    ind = np.argmax(count_array)
    coords = label == ind
    # the pixels corresponding to the most label is set to be 0, and rest pixels are set to be 1
    np_array[coords] = 0

    return np_array


# the mask refine algorithm that consist of multiple steps
def MultiStep_MaskRefine(ndwi):
    r"""
    :param ndwi: it is the WLM obtained by water-land segmentation algorithm -> np::ndarray -> [h w]
    :return:
    mask: updated WLM -> np::ndarray -> [h w]
    """
    ndwi = ndwi.astype(np.uint8)
    ndwi = 1 - ndwi
    # segment the mask into regions that are not isolated
    n_objects, labels = cv2.connectedComponents(ndwi)
    # the PrimElement is to segment the labels that have n_objects into two class, 0 for the most label and 1 for rests
    prim_class = PrimElement(labels, n_objects)
    prim_class = prim_class.astype(np.uint8)
    kernel = np.ones([7, 7], np.uint8)  # it must be odd
    closed_mat = cv2.morphologyEx(prim_class, cv2.MORPH_CLOSE, kernel)  # close function
    closed_mat = 1 - closed_mat
    # another connection analysis
    n_objects_, labels_ = cv2.connectedComponents(closed_mat)
    # added with PrimElement function
    second_clf = PrimElement(labels_, n_objects_)
    # the coords that corresponding to water (as line 203, it is reversed, so 1 is for water)
    water_region = ndwi == 1
    # use the water_region to mask the second_clf to obtain the mean value of these pixels, to find the water
    # corresponding to which label
    stat = np.mean(second_clf[water_region])
    ind_for_water = None
    # if the mean > 0.5, then it is 1; else, 0
    if stat > 0.5:
        ind_for_water = 1
    else:
        ind_for_water = 0

    ind_for_land = 1 - ind_for_water
    masked_zone = second_clf == ind_for_land
    mask = np.zeros([ndwi.shape[0], ndwi.shape[1]])
    mask[masked_zone] = 1

    # 1 for land and 0 for water
    mask = mask.astype(np.uint8)

    return mask


# water-land segmentation based on binary segmentation on B8
def B8mask(rhos, thres):
    r"""
    :param rhos: the NIR raster, for Sentinel-2 it is B8 (10m)
    :param thres: the threshold for binary segmentation
    :return:
    b8mask: the WLM
    b8mask_dt: the DtCM based on WLM
    """
    b8 = rhos[:, :, 7]
    b8mask = Binary_Threshold(b8, thres, 0, 1)
    # here the MultiStep_MaskRefine
    refine_b8mask = MultiStep_MaskRefine(b8mask)
    # refine_b8mask = cv2.medianBlur(refine_b8mask, 3)
    reserve_refine_b8mask = 1 - refine_b8mask
    reserve_refine_b8mask = reserve_refine_b8mask.astype(np.uint8)
    b8mask_dt = cv2.distanceTransform(reserve_refine_b8mask, distanceType=cv2.DIST_L2, maskSize=5)

    return b8mask, b8mask_dt


# the main function, it read the rhos and its corresponding dem
def load_rhos_gt(path, train_dbm_path, mask_path, nan_filter=True):
    r"""
    :param path: the path of rhos tif -> string
    :param train_dbm_path: the path of dem tif -> string
    :param mask_path: the path of the mask tif -> string
    :param nan_filter: do it perform filter on nan or invalid pixels -> bool
    :return:
    rhos_raster: the rhos raster obtained -> np::ndarray -> [h w c]
    elev_raster: the elev raster obtained -> np::ndarray -> [h w]
    extent: the extent of rhos(elev) tif -> gdal::GeoTransform
    proj: the proj of rhos(elev) tif -> gdal::Projection
    """
    rhos = gdal.Open(path)
    extent = rhos.GetGeoTransform()
    proj = rhos.GetProjection()

    dem = gdal.Open(train_dbm_path)
    dem_proj = dem.GetProjection()

    # elevation map
    tmp_raster = r'./tmp.tif'
    driver = gdal.GetDriverByName('GTiff')
    handler = driver.Create(tmp_raster, rhos.RasterXSize, rhos.RasterYSize, 1, gdal.GDT_Float32)
    handler.SetGeoTransform(extent)
    handler.SetProjection(proj)

    handler.GetRasterBand(1).SetNoDataValue(-111111)
    # perform re-projection for dem, to align with the rhos
    gdal.ReprojectImage(dem, handler, dem_proj, proj, gdalconst.GRA_Bilinear)
    elev_raster = handler.GetRasterBand(1).ReadAsArray()

    handler.FlushCache()
    handler = None

    os.remove(tmp_raster)
    # introduce the mask raster
    mask = gdal.Open(mask_path)
    mask_proj = mask.GetProjection()
    handler = driver.Create(tmp_raster, rhos.RasterXSize, rhos.RasterYSize, 1, gdal.GDT_Float32)
    handler.SetGeoTransform(extent)
    handler.SetProjection(proj)

    handler.GetRasterBand(1).SetNoDataValue(-111111)
    # re-project mask to align with rhos
    gdal.ReprojectImage(mask, handler, mask_proj, proj, gdalconst.GRA_Bilinear)
    mask_raster = handler.GetRasterBand(1).ReadAsArray()
    mask_ndv = handler.GetRasterBand(1).GetNoDataValue()
    # bool_mask is for determine the area where is needed to perform interpolation (improve efficiency).
    bool_mask = mask_raster != mask_ndv

    handler.FlushCache()
    handler = None

    os.remove(tmp_raster)

    # initiate the interpolated rhos raster
    rhos_raster = np.ones([rhos.RasterYSize, rhos.RasterXSize, rhos.RasterCount]) * -111111

    # channel by channel
    for idx in range(rhos.RasterCount):
        band = rhos.GetRasterBand(idx + 1).ReadAsArray()
        # nan_coords = np.where(np.isnan(band))
        ndv = rhos.GetRasterBand(idx + 1).GetNoDataValue()
        # determine the index of nan pixels
        ndvv = np.isnan(band)
        # determine the index of ndv pixels
        ndv_c = band == ndv
        # concatenate these two bool matrix
        nan_coords = np.concatenate([ndvv[:, :, None], ndv_c[:, :, None]], axis=-1)
        # get the compact bool matrix
        nan_coords = np.max(nan_coords, axis=-1)
        # mask the nan_coords by bool_mask (after this, only area that need to be interpolated is true,
        # others are false)
        nan_coords = nan_coords * bool_mask
        nan_coords = np.where(nan_coords == True)
        if nan_filter:  # if we need this interpolation process
            update_band = nan_interp(band, nan_coords, 0.25, ndv=ndv)
        else:  # if we do not need it, just copy the original matrix
            update_band = copy.deepcopy(band)
            update_band[nan_coords[0], nan_coords[1]] = -111111

        v_ndv_c = update_band == ndv
        update_band[v_ndv_c] = -111111
        rhos_raster[:, :, idx] = update_band

    return rhos_raster, elev_raster, extent, proj


def rhos_d_preprocess(raster_path, dem_path, mask_path, thres, n=500, nan_filter=True, is_output=True, PATH=None):
    r"""
    :param raster_path: the path for the rhos raster -> string
    :param dem_path: the path for the dem raster -> string
    :param mask_path: the path for the mask raster -> string
    :param thres: the threshold value for the b8mask function
    :param n: the top n elevations in the processed dem are performed MAD outlier detection program -> int
    :param nan_filter: perform interpolation for invalid pixel -> bool
    :param is_output: do the processed data output as a file -> bool
    :param PATH: the path for the output of processed data, if it is None, its path is the directory of raster -> string
    :return:
    """
    rhos, elev, extent, proj = load_rhos_gt(raster_path, dem_path, mask_path, nan_filter=nan_filter)
    # get the file name of raster, for further output
    update_rhos_path = raster_path.split('.')[0]
    # the valid_rhos is used to construct a bool mask for invalid pixels, if there is an abnormal value in one pixel,
    # then this pixel will be marked as invalid
    valid_rhos = np.zeros([rhos.shape[0], rhos.shape[1]])
    # channel by channel, to find the invalid pixels
    for iidx in range(rhos.shape[-1]):
        valid_rhos += rhos[:, :, iidx] > 0
    valid_rhos = valid_rhos // rhos.shape[-1]  # only the pixel of 1 is completely valid for every channel
    valid_rhos = valid_rhos.astype(bool)
    # convert to bool matrix
    nvd_coords = (1 - valid_rhos).astype(bool)
    # set the non-valid pixels in elevations as -111111, which is used to label invalid pixels
    elev[nvd_coords] = -111111

    # perform binary threshold algorithm to obtain b8mask and b8mask_dt
    b8mask, b8mask_dt = B8mask(rhos, thres)
    # correct the coastal line dems by CORRECT_DBM function, obtain the updated elev and WLM mask
    elev, update_b8mask = CORRECT_DBM(b8mask_dt, b8mask, elev)
    # another MultiStep_MaskRefine is applied, since the WLM has been updated in above process
    update_b8mask = MultiStep_MaskRefine(update_b8mask)
    # get the DtCM of update_b8mask
    reverse_b8mask = (1 - update_b8mask).astype(np.uint8)
    update_dt = cv2.distanceTransform(reverse_b8mask, distanceType=cv2.DIST_L2, maskSize=5)

    # obtained the update_dem (elev that have been updated) by the mask of update_b8mask
    update_dem = np.ones([elev.shape[0], elev.shape[1]]) * -111111
    update_dem[update_b8mask == 0] = elev[update_b8mask == 0]

    # get the valid gts, as a sequence
    valid_gts = update_dem[update_dem != -111111]

    # sort the valid_gts
    valid_gts_sorted = np.sort(valid_gts)
    # reverse the valid_gts_sorted and get the top n samples
    valid_gts_sorted = valid_gts_sorted[::-1]
    highest_groups = valid_gts_sorted[:n]
    # construct a temp coordinates
    coords = [np.zeros([n]), np.arange(n)]

    # use the MAD to detect outliers
    remove_coords, median, mad = MAD(highest_groups, coords, 3)

    for idx in range(remove_coords[1].shape[0]):
        # these indexes are used to set these values into -111111 (ndv)
        r_idx = remove_coords[1][idx]
        highest_groups[r_idx] = -111111

    # get the max elevation
    max_elev = np.max(highest_groups)
    update_dem[update_dem > max_elev] = -111111
    # convert to bathymetry -> Hmax - elevation
    update_dem[update_dem != -111111] = max_elev - update_dem[update_dem != -111111]
    # get the bathymetry
    dbm = copy.deepcopy(update_dem)
    del update_dem
    print("The data have processed, and the maximum elevation selected is: %.4f" % max_elev)

    # if these processed raster needed to output, if output, it will add with "_RMDB_MAXE_XXXXX(max elevation).tif"
    if is_output:
        if PATH is None:
            update_rhos_path = update_rhos_path + '_RMDB_MAXE_' + str(round(max_elev, 4)) + '.tif'
        else:
            if not os.path.exists(PATH):
                os.mkdir(PATH)
            update_rhos_path = PATH + '/' + update_rhos_path.split('/')[-1] + '_RMDB_MAXE_' + \
                               str(round(max_elev, 4)) + '.tif'
        # this data contain [R: rhos M: mask, D: DtCM, B: Bathymetry] -> RMDB
        raster_output(update_rhos_path, [rhos, update_b8mask, update_dt, dbm], extent=extent, proj=proj)

    else:
        return rhos, update_b8mask, update_dt, dbm, extent, proj


# the function for fix in-valid pixels in rhos.
def invalid_pixels_fixed(rhos, mask, ndv):
    num_channels = rhos.shape[-1]
    # here, we select N samples
    N = 100000
    B8 = rhos[:, :, 7]  # get the B8
    selected_rhos = B8[B8 > 0]
    selected_rhos = np.sort(selected_rhos)[N - 1]
    candidate_coords = (B8 <= selected_rhos) * (B8 > 0)
    mean_v = np.ones([num_channels, ], dtype=np.float32)

    # channel by channel
    for idx in range(num_channels):
        band = rhos[:, :, idx]
        # determine the index of nan and ndv pixels
        nan_c = np.isnan(band)
        ndv_c = band == ndv
        # concatenate these two bool matrix
        nan_coords = np.concatenate([nan_c[:, :, None], ndv_c[:, :, None]], axis=-1)
        # get the compact bool matrix
        nan_coords = np.max(nan_coords, axis=-1)
        # mask the nan_coords by bool_mask (after this, only area that need to be interpolated is true,
        # others are false)
        invalid_coords = nan_coords * mask
        invalid_coords = np.where(invalid_coords is True)
        # interpolate the invalid pixels located in the mask region
        band = nan_interp(band, invalid_coords, 0.25, ndv=ndv)

        # interpolate the pixels except mask
        invalid_mask_out = nan_coords * (1 - mask).astype(bool)
        interpolated_v = np.mean(band[candidate_coords])
        band[invalid_mask_out] = interpolated_v
        mean_v[idx] = interpolated_v
        rhos[:, :, idx] = band

    return rhos, mean_v
