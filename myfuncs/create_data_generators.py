# create_data_generators.py
# Function to create data generators

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(input_shape, batch_size, train_dir, test_dir):
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen_test = ImageDataGenerator(rescale=1./255)

    generator_train = datagen_train.flow_from_directory(
        directory=train_dir,
        target_size=input_shape,
        batch_size=batch_size,
        shuffle=True
    )

    generator_test = datagen_test.flow_from_directory(
        directory=test_dir,
        target_size=input_shape,
        batch_size=batch_size,
        shuffle=False
    )

    return generator_train, generator_test