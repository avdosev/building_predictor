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
import config
from common import train_pipe, test_pipe, bfs

def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, :, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :, :]
    return image

def augment(image):
    image = horizontal_flip(image)
    image = vertical_flip(image)
    return image



class Maps(keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # получаем все пути к снимкам
        city_paths = [os.path.join(root, file) for root, _, files in os.walk('data/train') if len(files) > 0 for file in files][6:7]
        # загружаем все в память
        y = []
        x = []
        x2 = []
        print('start preparing')
        for city_path in city_paths:
            print(f'preparing "{city_path}"')
            df = gdal.Open(city_path)
            data = df.GetRasterBand(1).ReadAsArray()
            for i in range(0, data.shape[0]-11, 7):
                for j in range(0, data.shape[1]-11, 3):
                    val = data[i+5,j+5]
                    
                    # need skip
                    if val == 0 or (val == 2 and i % 2 == 1):
                        continue
                    
                    x.append(np.expand_dims(data[i:i+11,j:j+11], axis=2))
                    x2.append(bfs(i+5,j+5,data))
                    y.append(val)
        
        y = np.array(y)
        y = test_pipe(y)
        
        x = np.array(x)
        x = train_pipe(x)
        
        print('input shape:', x.shape)
        print('output shape:', y.shape)
        print('preparation ready')
        self.y = y
        self.x = x
        self.x2 = x2

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = np.array(self.x[idx * self.batch_size:
                              (idx + 1) * self.batch_size])
        batch_x2 = np.array(self.x2[idx * self.batch_size:
                              (idx + 1) * self.batch_size])

        # batch_x = augment(batch_x)
        
        batch_y = np.array(self.y[idx * self.batch_size:
                              (idx + 1) * self.batch_size])
        
        return [batch_x, batch_x2], batch_y

def main():
    name = 'first'
    model_path = f'models/{name}_latest.hdf5'

    model = m.get_model(4)

    # if os.path.exists(model_path):
    #     model.load_weights(model_path)

    model.summary()

    optimizer = keras.optimizers.Adam(lr=0.001)


    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'mae', tf.keras.metrics.FalseNegatives(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

    train_dataset = Maps(config.batch_size)
    model.fit(
        train_dataset,
        epochs=20,
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
            keras.callbacks.ModelCheckpoint(
                filepath=f'models/model_min_{name}.hdf5',
                save_weights_only=True,
                monitor='false_negatives',
                mode='min',
                save_best_only=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f'models/model_min_mae_{name}.hdf5',
                save_weights_only=True,
                monitor='mae',
                mode='min',
                save_best_only=True
            ),
            # keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        ]
    )

    model.save(model_path)


if __name__ == '__main__':
    main()