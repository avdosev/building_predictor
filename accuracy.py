from sklearn.metrics import accuracy_score
from osgeo import gdal 

dt = gdal.Open('data/city/belgrad/90.tif')
dt2 = gdal.Open('data/output/belgrad_predict.tif')

bands = dt.GetRasterBand(1).ReadAsArray() [1000+5:3000-5, 5:2500-5]
bands2 = dt2.GetRasterBand(1).ReadAsArray()

print('score:', accuracy_score(bands.ravel(), bands2.ravel()))
print('original 3:', len(bands[bands == 3]))
print('predicted 3:', len(bands[bands2 == 3]))