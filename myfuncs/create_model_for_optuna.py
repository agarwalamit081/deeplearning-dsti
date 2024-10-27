# create_model_for_optuna.py
# Create a model for Optuna

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from myfuncs.create_model import create_model
from myfuncs.metrics import metrics

def create_model_for_optuna(params):
    '''
    Function to create a model for Optuna using dynamic hyperparameters.
    '''
    # Pass the entire params dictionary as hyperparameters
    model = create_model(
        input_shape=(224, 224, 3),
        num_classes=196,
        hyperparams=params  # Pass the params dictionary instead of individual arguments
    )
    
    # Compile the model with custom metrics
    model.compile(
        optimizer=model.optimizer,  # Already compiled within create_model
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    return model


