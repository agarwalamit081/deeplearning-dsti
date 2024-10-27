# se_block.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Reshape
from tensorflow.keras import layers

class SEBlock(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense_1 = Dense(channels // self.reduction, activation='relu')
        self.dense_2 = Dense(channels, activation='sigmoid')
        self.reshape = Reshape([1, 1, channels])
        self.multiply = Multiply()

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.dense_1(se)
        se = self.dense_2(se)
        se = self.reshape(se)
        return self.multiply([inputs, se])

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({"reduction": self.reduction})
        return config

