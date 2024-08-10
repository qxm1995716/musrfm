Here, the data preprocessing can be roughly devided into two parts, **Sentinel-2 L2A data processing** and **conversion from DEM to DBM**. The code for it can be found in the ./data_reader.py. <br><br>

In our code, we use the **rhos_d_preprocess** function in basic_modules.py to process the raw L2A data and then save them for further usage.  This function contains both **Sentinel-2 L2A data processing** and **conversion from DEM to DBM** two parts. <br><br>
1. rhos_d_preprocess function
```
def rhos_d_preprocess(raster_path, dem_path, mask_path, thres, n=500, nan_filter=True, is_output=True, PATH=None):
```
This function has 8 parameters that need to be manual set, including: <br>
- raster_path: the path of the Sentinel-2 L2A imagery raster. <br>
- dem_path: the path of the DEM raster. <br>
- mask_path: the path of the mask raster that used to highligh the area needed to be interpolated to fill invalid pixels. <br>
- thres: the threshold that needed to segment the B8 band into water and land. <br>
- n: the number of samples that used to perform MAD to exclude abnormal pixels before transfer elevation into bathymetry. <br> 
```
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
```
