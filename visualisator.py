from osgeo import gdal 
import matplotlib.pyplot as plt 

dt = gdal.Open('data/city/nizniy/NN.tif')

print(dt.RasterCount)

bands = dt.GetRasterBand(1).ReadAsArray()

bands = bands[:10000, :10000]

f = plt.figure() 
plt.imshow(bands) 
plt.savefig('d3.png') 