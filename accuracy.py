from osgeo import gdal 
import numpy as np

dt = gdal.Open('data/test/belgrad/90.tif')
dt2 = gdal.Open('data/output/belgrad_predict.tif')

bands = dt.GetRasterBand(1).ReadAsArray() [1000+5:3000-5, 5:2500-5]
bands2 = dt2.GetRasterBand(1).ReadAsArray()

print('orig shape:', bands.shape)
print('predict shape:', bands2.shape)

print('originals 3:', np.sum(bands == 3))
print('predicted 3:', np.sum(bands2 == 3))

TP = np.sum((bands == 3) & (bands2 == 3))
TN = np.sum((bands != 3) & (bands2 != 3))
FN = np.sum((bands == 3) & (bands2 != 3))
FP = np.sum((bands != 3) & (bands2 == 3))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
