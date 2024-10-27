# metrics.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Precision, Recall, AUC

# Define the metrics to be used in the models
metrics = [
    'categorical_accuracy',
    'accuracy',
    TopKCategoricalAccuracy(k=3,name="top_3_acc"),
    TopKCategoricalAccuracy(k=5,name="top_5_acc"),
    Precision(name='precision'), 
    Recall(name='recall'),
    AUC(name='auc')
]


