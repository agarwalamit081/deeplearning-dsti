# gradient_accumulation.py
# GradientAccumulation

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

import tensorflow as tf

class GradientAccumulation(tf.keras.callbacks.Callback):
    def __init__(self, accum_steps=4):
        self.accum_steps = accum_steps
        self.batch = 0
        self.accum_grads = None

    def on_train_batch_begin(self, batch, logs=None):
        if self.batch % self.accum_steps == 0:
            self.accum_grads = [tf.zeros_like(var) for var in self.model.trainable_weights]
            self.model.optimizer.get_gradients = self._get_gradients

    def _get_gradients(self, loss, params):
        grads = super(self.model.optimizer.__class__, self.model.optimizer).get_gradients(loss, params)
        self.accum_grads = [accum_grad + grad for accum_grad, grad in zip(self.accum_grads, grads)]
        return self.accum_grads

    def on_train_batch_end(self, batch, logs=None):
        if self.batch % self.accum_steps == 0:
            self.model.optimizer.apply_gradients(zip(self.accum_grads, self.model.trainable_weights))
            self.accum_grads = None

        self.batch += 1
