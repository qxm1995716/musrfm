The workflow of Sentinel-2 L2A fusion and download. In our research, we use GEE to obtain the median fusion of multiple L2A imageries and export it. Since the data used is very large, so it diffcult to upload our data, so we provide this workflow for anyone who want create their own dataset.

The requirements are as follows: <br>
(1) The Google account to log in GEE platform.  <br>
(2) Target area coordinates, or you can create a new layer and draw the rectangle area.  <br>
(3) The threshold for cloud, defalut is 10 and you can adapt it to get the most staisfied fused imagery.   <br>

In **./gee script/**, we provide a templete to create and export the fused L2A imagery to the google drive, you can modify it and get your own data.
