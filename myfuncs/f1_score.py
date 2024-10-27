# f1_score.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

# Instantiate these once to avoid creating them inside a tf.function.
precision_metric = Precision()
recall_metric = Recall()

def f1_score(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))