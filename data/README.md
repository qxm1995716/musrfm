The workflow of Sentinel-2 L2A fusion and download. In our research, we use GEE to obtain the median fusion of multiple L2A imageries and export it. Since the data used is very large, so it diffcult to upload our data, so we provide this workflow for anyone who want create their own dataset.

The required data as follows: <br>
(1) The Google account to log in GEE platform.  <br>
(2) The coordinate of the target area.   <br>
(3) Some key parameters, including:   <br>
[a] Target area coordinates, or you can create a new layer and draw the rectangle area;   <br>
[b] The threshold for cloud, defalut is 10 and you can adapt it to get the most staisfied fused imagery.   <br>
