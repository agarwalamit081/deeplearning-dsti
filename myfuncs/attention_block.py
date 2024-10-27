# attention_block.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Reshape
from tensorflow.keras import layers

class AttentionBlock(layers.Layer):
    '''
    The function adds attenuation block to the layers of the neural network.
    Inputs:
        The layer of the neural networks is given as the input
    Output:
    The function returns the configuration for the given layer.
    '''
    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense_1 = Dense(channels // 8, activation='relu')
        self.dense_2 = Dense(channels, activation='sigmoid')
        self.reshape = Reshape([1, 1, channels])
        self.multiply = Multiply()

    def call(self, inputs):
        attention = self.global_avg_pool(inputs)
        attention = self.dense_1(attention)
        attention = self.dense_2(attention)
        attention = self.reshape(attention)
        return self.multiply([inputs, attention])

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        return config
