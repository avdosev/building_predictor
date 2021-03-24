import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
from osgeo import gdal

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow import keras
import tensorflow as tf
import math
import model as m

def train_pipe(val):
    # 0 = no data
    # 1 = water surface
    # 2 = land no built-up in any epoch
    # 3 = built-up
    i = np.where(val == 3, 2, val)
    i = np.where(i > 3, 3, val)
    return i

def test_pipe(data):
    i = np.where(data > 3, 3, data)
    res = np.zeros((i.shape[0], 4), dtype=bool)
    for j, item in enumerate(i):
        res[j, item] = True
    return res

class Maps(keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # получаем все пути к снимкам
        city_paths = [os.path.join(root, file) for root, _, files in os.walk('data/train') if len(files) > 0 for file in files]
        # загружаем все в память
        y = []
        x = []
        print('start preparing')
        for city_path in city_paths:
            print(f'preparing "{city_path}"')
            df = gdal.Open(city_path)
            data = df.GetRasterBand(1).ReadAsArray()
            for i in range(0, data.shape[0]-11, 10):
                for j in range(0, data.shape[1]-11, 10):
                    x.append(np.expand_dims(data[i:i+11,j:j+11], axis=2))
                    y.append(data[i+5,j+5])
        
        y = np.array(y)
        y = test_pipe(y)
        
        x = np.array(x)
        x = train_pipe(x)
        
        print('input shape:', x.shape)
        print('output shape:', y.shape)
        print('preparation ready')
        self.y = y
        self.x = x

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = np.array(self.x[idx * self.batch_size:
                              (idx + 1) * self.batch_size])
        batch_y = np.array(self.y[idx * self.batch_size:
                              (idx + 1) * self.batch_size])
        
        return batch_x, batch_y

def main():
    train_dataset = Maps(5000)
    name = 'first'
    model_path = f'models/{name}_latest.hdf5'

    model = m.get_model(4)

    if os.path.exists(model_path):
        model.load_weights(model_path)

    model.summary()

    optimizer = keras.optimizers.Adam(lr=0.001)


    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'mae'])

    model.fit(
        train_dataset,
        epochs=10,
        initial_epoch=0,
        callbacks=[
            # keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=4, verbose=0, mode="min"),
            keras.callbacks.ModelCheckpoint(
                filepath=f'models/model_best_{name}.hdf5',
                save_weights_only=True,
                monitor='accuracy',
                mode='max',
                save_best_only=True
            ),
            # keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        ]
    )

    model.save(model_path)


if __name__ == '__main__':
    main()