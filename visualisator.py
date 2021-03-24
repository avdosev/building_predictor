from osgeo import gdal 
import matplotlib.pyplot as plt 

dt = gdal.Open('data/test/belgrad/90.tif')
# dt = gdal.Open('data/output/belgrad_predict.tif')

print(dt.RasterCount)

bands = dt.GetRasterBand(1).ReadAsArray() [1000+5:3000-5, 5:2500-5]
bands = dt.GetRasterBand(1).ReadAsArray()

f = plt.figure() 
plt.imshow(bands, cmap='hot')
plt.colorbar() 
plt.savefig('d3.png') 