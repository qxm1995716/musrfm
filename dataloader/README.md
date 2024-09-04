# Step 1 <br>
Since the obtained Sentinel-2 L2A imagery from GEE can have some invalid pixels, here we need process it and combine it with the obtained DBM, forming a raster with shape H * W * (C + 3). The C is the channels of L2A imagery, the default value of it is 12, since we include all available bands, namely B1, B2, B3, B4, B5, B6, B7, B8, B8A, B10, B11, B12. The obtained data should be stored and for further usage. 

This step is finished by **rhos_d_preprocess** function in **basic_modules.py**, and its detailed explaination is located in the docs.md. Here, we provide a templete for process a L2A imagery and its corresponding DEM data. <br><br>
**HIGHLIGHTS: In our code, the nondata value is -111111, please keep mind on this, especially when create raster of mask or dem, to avoid any unpredicted bugs.**

```
raster_path = './l2a_imagery.tif'  # the input L2A imagery data.
dem_path = './dem_data.tif'  # the input DEM data.
mask_path = './mask_path.tif'  # the mask for the area that need bilinear interpolation function to inpaint the invalid pixels
# for other parameters, you can find their meaning and function in docs.md. 
rhos_d_preprocess(raster_path, dem_path, mask_path, thres, n=500, nan_filter=True, is_output=True, PATH=None)
# after this, you can find the output raster in the PATH dict, if the is_output is False, this function will return processed data, which you can found in the code. 
```

# Step 2 <br>
Then, the **DataReadIn** function in data_reader.py is used to process the obtained raster above, and perform further process. This function is used before the DataLoader function in the training.py or fine-tuning.py, and its defination shown as follows:
```
DataReadIn(path_list, process_type, c_num=12, tmp_dict=None, is_dbm=True, max_bathy=25, stride=10, is_random_shift=True,
           random_shift_scale=4, patch_size=15, mb_res=np.array([10, 30, 90, 270, 810]), basic_res=10, data_dict=None) 
```
