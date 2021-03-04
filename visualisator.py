from osgeo import gdal 
import matplotlib.pyplot as plt 

dt = gdal.Open('d3.tif')

print(dt.RasterCount)

bands = dt.GetRasterBand(1).ReadAsArray()

bands = bands[:2000, :2000]

f = plt.figure() 
plt.imshow(bands) 
plt.savefig('d3.png') 
plt.show() 