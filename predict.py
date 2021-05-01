
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
from common import train_pipe, find_info
import config
import numba

@numba.njit()
def split_map(data):
    res = []
    for i in range(0, data.shape[0]-config.map_size+1): 
        for j in range(0, data.shape[1]-config.map_size+1):
            res.append(np.expand_dims(data[i:i+config.map_size,j:j+config.map_size], axis=2))
    return res

@numba.njit()
def split_find_info(data):
    y_shape = data.shape[0]-config.map_size+1
    x_shape = data.shape[1]-config.map_size+1
    res = np.empty((y_shape*x_shape, 4+3))

    for i in numba.prange(0, y_shape):
        for j in range(0, x_shape):
            find_res = find_info(i+config.map_size//2, j+config.map_size//2, data)
            for k, f_r in enumerate(find_res):
                res[i*y_shape+j, k] = f_r

    return res

def process_city(filename, output_name, bounds):
    print('open data')
    dt = gdal.Open(filename)
    original_data = dt.GetRasterBand(1).ReadAsArray()
    print('original shape:', original_data.shape)
    if bounds is not None:
        original_data = original_data[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    data = train_pipe(original_data)
    print('open model')
    model = m.get_model(4)
    model.load_weights('models/model_best_first.hdf5')

    print('splitting...')
    splitted = np.array(split_map(data))
    print('find info...')
    info = np.array(split_find_info(original_data))

    print(splitted.shape)
    print(info.shape)

    print('predicting...')
    res = model.predict([splitted, info], verbose=True, batch_size=config.batch_size)

    print('preparing results')
    print('original shape', res.shape)
    print(data.shape[:2])
    transformed = np.reshape(np.argmax(res, axis=1), tuple(np.array(data.shape[:2])-config.map_size+1))
    original_data = original_data[config.map_size//2:data.shape[0]-config.map_size//2, config.map_size//2:data.shape[1]-config.map_size//2]

    print(transformed.shape)
    print(original_data.shape)

    # удаляем оригинальные данные
    without_predict = np.where(original_data == 3, 2, original_data)
    # формируем результат; склеиваем считаем что данных которых не было это 3
    full_res = np.where((transformed == 3) & (without_predict == 2), 3, without_predict)

    print('save_results')

    write_tif(f'data/output/{output_name}_predict.tif', full_res)

# process_city('data/test/belgrad/90.tif', 'belgrad', (config.map_size-100,3000,0,2500))
process_city('data/test/tst/82.tif', '82', (0, 3000, 0, 3000))
# process_city('data/test/tst/85.tif', '85', (0, 3000, 0, 3000))
# process_city('data/train/ekb/82.tif', 'ekb', (0, 3000, 0, 3000))