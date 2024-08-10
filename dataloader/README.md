**DATA PROCESSING MODULES**

Here, the data preprocessing can be roughly devided into two parts, **Sentinel-2 L2A data processing** and **conversion from DEM to DBM**. The code for it can be found in the ./data_reader.py. <br>
```
active_region = reproject_aligned(element['rhos'], element['active_region'])
active_region = active_region == -111111
rhos, wlm, dt, dbm, extent, proj = read_rmdb(element['rhos'], c_num)
dbm[active_region] = -111111
mask = reproject_aligned(element['rhos'], element['mask'])
mask = mask != -111111
rhos, interpolated_v = invalid_pixels_fixed(rhos, mask, ndv=-111111.)  # get the filled rhos data
data = np.concatenate([rhos, wlm[:, :, None], dt[:, :, None]], axis=-1)  # added with other 2 channel
data = data.astype(np.float32)
data = padding_function_original(data, mask, self.radius, max_scale_factor=max_scale, interpolated_v=None)
```
