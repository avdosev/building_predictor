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

def train_pipe(name: str):
    pass

def result_pipe(name: str):
    pass

class Maps(keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    
    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:
                              (idx + 1) * self.batch_size]
        batch_x = np.array([train_pipe(name) for name in batch_images])
        batch_y = np.array([result_pipe(name) for name in batch_images])
        
        return batch_x, batch_y

def main():
    train_dataset = Maps(20)
    name = 'first'
    model_path = f'models/{name}_latest.hdf5'

    model = m.get_model()

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
                monitor='mean_squared_error',
                mode='min',
                save_best_only=True
            ),
            # keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        ]
    )

    model.save(model_path)


if __name__ == '__main__':
    main()