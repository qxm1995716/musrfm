# Code of MCHR<br>
**Defination and usage** <br>
Here, the S2R10_MCHR.cpp is the code of MCHR for Sentinel-2 10 m resultion input by using C++. We complie the MCHR code with use of **Boost C++ Lib**, and we provide the compiled .pyd file of MCHR (for win10 os) and .so file (for ubuntu os, specifically, ubuntu18.04). The python version for both complied files is py38. <br> <br>
The templete of the usage of this function is come from the ./dataloader/data_reader.py, shown as:
```
mchr_data_list, coords_list = S2R10_MCHR.S2R10_MCHR(rhos, coords, mb_res, radius, ms_ratio)
```
This function need 5 input parameters: <br>
*rhos*: the padded Sentinel-2 L2A imagery. <br>
*coords*: the coords of the center of each patch. <br>
*mb_res*: the resolutions of multiple branches, here is [10, 30, 90, 270, 810]. <br>
*radius*: the radius of the cropped patch, and the height and width of each patch are (2 * radius + 1). <br>
*ms_ratio*: the ratio of used threads to perform mchr, its range is (0, 1], the less the value, the less cpu source is used while longer time the sampler will cost. <br><br>
**How to use it**
For window os, the S2R10_MCHR.pyd file need to be set at the path *./Anaconda/envs/YourInterpreter/Lib/site-packages/*, and then you can use it in the code once this interpreter is activated. <br>
For ubuntu os, THE add the S2R10_MCHR.so to the dictory of main function, for example, for this repos, the path of S2R10_MCHR.so is ./S2R10_MCHR.so, as the same of training.py. <br> 
It should be noted that, the name of both files should not change, that is, S2R10_MCHR.**. <br>
**Complie it by yourself**<br> 
Since the python version is locked, if you want to use it in interpreter of different python version, you need complie it by yourself with the boost lib. Here we only provide the templete of complie it by gcc in ubuntu os, shown as follows:
```
g++ -shared -fPIC -O2 -I/usr/include/python3.8 -I/usr/include/boost S2R10_MCHR.cpp -o S2R10_MCHR.so -lboost_python38 -lboost_numpy38
```
Here, the *-I/usr/include/python3.8* is the path of installed python, it can also be found in the path of anaconda, and the *-I/usr/include/boost* is the location of the boost lib. <br> 
