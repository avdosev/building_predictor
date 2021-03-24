
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt 

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow import keras
import tensorflow as tf
import model as m
from train import train_pipe

print('open data')
dt = gdal.Open('data/city/ekb/82.tif')
original_data = dt.GetRasterBand(1).ReadAsArray()
data = train_pipe(original_data)
print('open model')
model = m.get_model(4)
model.load_weights('models/model_best_first.hdf5')
res = []
print('predicting...')
for i in range(0, data.shape[0]-11):
    for j in range(0, data.shape[1]-11):
        splitted = np.array([np.expand_dims(data[i:i+11,j:j+11], axis=2)])
        res.append(model.predict(splitted))

print('preparing results')
res = np.array(res)
transformed = np.reshape(np.argmax(res, axis=1), tuple(np.array(data.shape[:2])-10))
# transformed = np.argmax(res, axis=1)
original_data = original_data[5:data.shape[0]-5, 5:data.shape[1]-5]
data = data[5:data.shape[0]-5, 5:data.shape[1]-5]

print(transformed.shape)
print(original_data.shape)

full_res = np.where((transformed == 3) & (data != 3), 3, original_data)

print('save_results')

f = plt.figure()
plt.imshow(full_res) 
plt.savefig('ekb_predicted.png') 

plt.imshow(original_data)
plt.savefig('ekb_original.png')
