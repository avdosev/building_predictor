
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
from osgeo import gdal
from helper import write_tif

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow import keras
import tensorflow as tf
import model as m
from common import train_pipe, bfs
import config

print('open data')
dt = gdal.Open('data/test/belgrad/90.tif')
original_data = dt.GetRasterBand(1).ReadAsArray()[1000:3000, :2500]
data = train_pipe(original_data)
print('open model')
model = m.get_model(4)
model.load_weights('models/model_best_first.hdf5')

print('splitting...')
splitted = np.array([np.expand_dims(data[i:i+11,j:j+11], axis=2) for i in range(0, data.shape[0]-10) for j in range(0, data.shape[1]-10)])
info = np.array([bfs(i+5, j+5, original_data) for i in range(0, data.shape[0]-10) for j in range(0, data.shape[1]-10)])
print('predicting...')
res = model.predict([splitted, info], verbose=True, batch_size=config.batch_size)

print('preparing results')
print('original shape', res.shape)

transformed = np.reshape(np.argmax(res, axis=1), tuple(np.array(data.shape[:2])-10))
original_data = original_data[5:data.shape[0]-5, 5:data.shape[1]-5]

print(transformed.shape)
print(original_data.shape)

# удаляем оригинальные данные
without_predict = np.where(original_data == 3, 2, original_data)
# формируем результат; склеиваем считаем что данных которых не было это 3
full_res = np.where((transformed == 3) & (without_predict == 2), 3, without_predict)

print('save_results')

write_tif('data/output/belgrad_predict.tif', full_res)