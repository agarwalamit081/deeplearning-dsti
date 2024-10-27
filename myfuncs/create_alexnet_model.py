# create_alexnet_model.py
# Model 2 - AlexNet

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from myfuncs.metrics import metrics
from myfuncs.finetune_alexnet_model import finetune_alexnet_model

def create_alexnet_model(input_shape, num_classes, hyperparams, x_trainable='all'):
    """
    Create the AlexNet model using dynamic hyperparameters.
    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes for the output layer.
    :param hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).
    :param x_trainable: Specifies whether to train all layers or freeze some.
    """
    # Create the AlexNet model using finetune_alexnet_model
    model = finetune_alexnet_model(input_shape, num_classes, hyperparams, x_trainable)

    # Extract hyperparameters dynamically
    learning_rate = hyperparams.get('learning_rate', 1e-6)
    weight_decay = hyperparams.get('weight_decay', 1e-6)
    dropout = hyperparams.get('dropout', 0.5)
    beta_1 = hyperparams.get('beta_1', 0.9)
    beta_2 = hyperparams.get('beta_2', 0.999)
    decay = hyperparams.get('decay', 0.0)

    # Select optimizer dynamically based on hyperparameters
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')  # Default is 'Adam' if not provided

    if optimizer_type == 'Adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            decay=decay
        )
    elif optimizer_type == 'SGD':
        optimizer = SGD(
            learning_rate=learning_rate,
            momentum=hyperparams.get('momentum', 0.9),
            decay=decay,
            nesterov=True
        )
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(
            learning_rate=learning_rate,
            rho=hyperparams.get('rho', 0.9),
            decay=decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Compile the model with the dynamically selected optimizer and other hyperparameters
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    
    return model
