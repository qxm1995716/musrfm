from osgeo import gdal
import os

os.environ['PROJ_LIB'] = r'D:\Anaconda\envs\acolite\Lib\site-packages\pyproj\proj_dir\share\proj'


def raster_output(file_path, ordered_numpy_array, extent, proj, ndv=-111111):
    driver = gdal.GetDriverByName('GTiff')
    height, width = ordered_numpy_array[0].shape[0], ordered_numpy_array[0].shape[1]
    c = 0
    for idx in range(len(ordered_numpy_array)):
        shape_vec = ordered_numpy_array[idx].shape
        if len(shape_vec) == 2:
            c = c + 1
        else:
            c = c + shape_vec[-1]

    handler = driver.Create(file_path, width, height, c, gdal.GDT_Float32)
    handler.SetGeoTransform(extent)
    handler.SetProjection(proj)

    b_idx = 1

    for idx in range(len(ordered_numpy_array)):
        shape_vec = ordered_numpy_array[idx].shape
        if len(shape_vec) == 2:
            handler.GetRasterBand(b_idx).WriteArray(ordered_numpy_array[idx])
            handler.GetRasterBand(b_idx).SetNoDataValue(ndv)
            b_idx = b_idx + 1
        else:
            for c_idx in range(ordered_numpy_array[idx].shape[-1]):
                handler.GetRasterBand(b_idx).WriteArray(ordered_numpy_array[idx][:, :, c_idx])
                handler.GetRasterBand(b_idx).SetNoDataValue(ndv)
                b_idx = b_idx + 1

    handler = None

    return 'Finished.'
