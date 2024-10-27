# extract_history_data.py
# Extract histories for the different iterations 

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

def extract_history_data(histories):
    '''
    Extract the metrics from the training process
    '''
    extracted_histories = []
    for history in histories:
        extracted_histories.append({
            'train_accuracy': history.history['accuracy'],
            'test_accuracy': history.history['val_accuracy'],  # Renamed from 'val_accuracy'
            'train_loss': history.history['loss'],
            'test_loss': history.history['val_loss']  # Renamed from 'val_loss'
        })
    return extracted_histories
