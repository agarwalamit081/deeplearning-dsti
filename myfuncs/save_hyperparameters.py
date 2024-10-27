# save_hyperparameters.py
# Save hyperparameters

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

import os
import pickle


def save_hyperparameters(model_name, params):
    '''
    A function to save the optimal hyperparameters to a file
    '''
    with open(f'{model_name}_hyperparameters.pkl', 'wb') as f:
        pickle.dump(params, f)
