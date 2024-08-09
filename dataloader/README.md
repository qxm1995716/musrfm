Here, the data preprocessing can be roughly devided into two parts, **Sentinel-2 L2A data processing** and **conversion from DEM to DBM**.

**1. Sentinel-2 L2A data processing** <br>
The median filtering of multi-temporal series remote sensing imageries in this part can be obtained through GEE using the script code in /data/get_fused_sr_data. Since there will be some non-value pixels ​​in the obtained fused sr imagery, these invalid pixels ​​must be properly processed, otherwise the inversion results will be abnormal. However, filling all the invalid pixels in the entire imagery may be a time-consuming job, we use a mask here to determine the area where the interpolation is performed, so a shapefile will be needed here as the mask.
