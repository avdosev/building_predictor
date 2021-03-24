from osgeo import gdal

def write_tif(dst_filename, arr):
    x_pixels = arr.shape[1] # number of pixels in x
    y_pixels = arr.shape[0] # number of pixels in y
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
           dst_filename,
           x_pixels,
           y_pixels,
           1,
           gdal.GDT_Byte, )
    dataset.GetRasterBand(1).WriteArray(arr)
    dataset.FlushCache()  # Write to disk.