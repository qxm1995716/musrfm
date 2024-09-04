# Step 1 <br>
Since the obtained Sentinel-2 L2A imagery from GEE can have some invalid pixels, here we need process it and combine it with the obtained DBM, forming a raster with shape H * W * (C + 3). The C is the channels of L2A imagery, the default value of it is 12, since we include all available bands, namely B1, B2, B3, B4, B5, B6, B7, B8, B8A, B10, B11, B12. The obtained data should be stored and for further usage. 

This step is finished by **rhos_d_preprocess** function in **basic_modules.py**, and its detailed explaination is located in the docs.md. Here, we provide a templete for process a L2A imagery and its corresponding DEM data. <br><br>
**HIGHLIGHTS: In our code, the nondata value is -111111.0, please keep mind on this, especially when create raster of mask or dem, to avoid any unpredicted bugs.**

```
raster_path = './l2a_imagery.tif'  # the input L2A imagery data. 
dem_path = './dem_data.tif'  # the input DEM data.
mask_path = './mask_path.tif'  # the mask for the area that need bilinear interpolation function to inpaint the invalid pixels
# for other parameters, you can find their meaning and function in docs.md. 
rhos_d_preprocess(raster_path, dem_path, mask_path, thres, n=500, nan_filter=True, is_output=True, PATH=None)
# after this, you can find the output raster in the PATH dict, if the is_output is False, this function will return processed data, which you can found in the code. 
```

# Step 2 <br>
Then, the **DataReadIn** function in data_reader.py is used to process the obtained raster above, and perform further process, and the most importantly, the multiple resulotion scale patches obtained by MCHR. This function is used before the DataLoader function in the training.py or fine-tuning.py, and its defination shown as follows:
```
DataReadIn(path_list, process_type, c_num=12, tmp_dict=None, is_dbm=True, max_bathy=25, stride=10, is_random_shift=True,
           random_shift_scale=4, patch_size=15, mb_res=np.array([10, 30, 90, 270, 810]), basic_res=10, data_dict=None) 
```
We take the code of loading training dataset as a example to explain this function. 
```
trains_container = DataReadIn(train_files, process_type='RMDB', c_num=12, is_dbm=True, is_random_shift=False,            
                              random_shift_scale=args.random_shift_scale, tmp_dict=tmp_train_path, stride=args.stride,
                              max_bathy=args.max_depth, patch_size=args.patch_size, mb_res=args.mb_res, basic_res=10, 
                              data_dict=tmp_train_path)
```
Here, the means of various parameters are as follows. <br>
- *train_files*: the rasters that processed by step 1. <br> 
- *process_type*: a flag for data, here we only use 'RMDB', which means that the data contained in the raster are [Reflectances (R), Mask of water-land (M), Distance to coast (D), Bathytmery (B)]. <br>
- *c_num*: the number of channels of reflectances. <br>
- *is_dbm*: a bool value to indicate that whether the DBM is included in the input rasters. <br>
- *is_random_shift*: whether random shift the coordinate of centeral point of each patch where perform MCHR.<br>
- *random_shift_scale*: the scale for random shift of central coordinates. <br>
- *tmp_dict*: the temporary dict to save padded raster, this is to avoid the MCHR exceeding the edge of input raster, while the scale is depend on the max range of MCHR. <br>
- *stride*: the stride step for the window shift. <br>
- *max_bathy*: the maximum bathymetry value. <br>
- *patch_size*: the shape of the patch of MCHR. <br>
- *mb_res*: the resolution of all branches. <br>
- *basic_res*: the resolution of input multi-spectral imagery. <br>
- *data_dict*: actually this is the tmp_dict. <br><br>

# Step 3 <br>
The dataloader 
