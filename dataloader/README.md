Step 1: 
Since the obtained Sentinel-2 L2A imagery from GEE can have some invalid pixels, here we need process it and combine it with the obtained DBM, forming a tensor with shape H * W * (C + 3). The C is the channels of L2A imagery, the default value of it is 12, since we include all available bands, namely B1, B2, B3, B4, B5, B6, B7, B8, B8A, B10, B11, B12. The obtained data should be stored and for further usage. 

In this code, processing  
