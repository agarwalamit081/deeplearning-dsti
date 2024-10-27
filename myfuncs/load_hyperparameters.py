# load_hyperparameters.py
# Load hyperparameters

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

import os
import pickle


def load_hyperparameters(model_name):
    '''
    Function to load existing hyperparameters, if any
    '''
    try:
        with open(f'{model_name}_hyperparameters.pkl', 'rb') as f:
            params = pickle.load(f)
        return params
    except FileNotFoundError:
        return None
