# custom_layer.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Multiply, Reshape

class CustomLayer(Layer):
    def __init__(self, block_type='se', reduction=16, **kwargs):
        """
        Custom layer that can function as both an SE block and an Attention block.
        :param block_type: 'se' for Squeeze-and-Excitation block, 'attention' for Attention block
        :param reduction: Reduction ratio for SE block
        """
        super(CustomLayer, self).__init__(**kwargs)
        self.block_type = block_type
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        if self.block_type == 'se':
            self.global_avg_pool = GlobalAveragePooling2D()
            self.dense_1 = Dense(channels // self.reduction, activation='relu')
            self.dense_2 = Dense(channels, activation='sigmoid')
        elif self.block_type == 'attention':
            self.global_avg_pool = GlobalAveragePooling2D()
            self.dense_1 = Dense(channels // 8, activation='relu')
            self.dense_2 = Dense(channels, activation='sigmoid')

        self.reshape = Reshape([1, 1, channels])
        self.multiply = Multiply()

    def call(self, inputs):
        attention_or_se = self.global_avg_pool(inputs)
        attention_or_se = self.dense_1(attention_or_se)
        attention_or_se = self.dense_2(attention_or_se)
        attention_or_se = self.reshape(attention_or_se)
        return self.multiply([inputs, attention_or_se])

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({
            "block_type": self.block_type,
            "reduction": self.reduction
        })
        return config


