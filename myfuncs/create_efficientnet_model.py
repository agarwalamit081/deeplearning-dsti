# create_efficientnet_model.py
# Model 7 - EfficientNet

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from myfuncs.finetune_efficientnet_model import finetune_efficientnet_model
from myfuncs.metrics import metrics

def create_efficientnet_model(input_shape, num_classes, hyperparams):
    """
    Create EfficientNet model with dynamic hyperparameters.
    
    Args:
    - input_shape: Shape of the input data (e.g., (224, 224, 3)).
    - num_classes: Number of output classes.
    - hyperparams: Dictionary of hyperparameters (default_hyperparams or best_params).
    """

    # Load the base EfficientNetB0 model without the top classification layers
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freezing layers conditionally based on x_trainable value
    x_trainable = hyperparams.get('x_trainable', 'all')
    if x_trainable == 'partial':
        for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4 layers
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True if x_trainable == 'all' else False

    # Call finetune_efficientnet_model with dynamic hyperparameters
    model = finetune_efficientnet_model(base_model, num_classes, hyperparams)

    # Select the optimizer based on hyperparams['optimizer_type']
    if hyperparams['optimizer_type'] == 'Adam':
        optimizer = Adam(
            learning_rate=hyperparams.get('learning_rate', 1e-6),
            beta_1=hyperparams.get('beta_1', 0.9),
            beta_2=hyperparams.get('beta_2', 0.999),
            decay=hyperparams.get('decay', 0.0)
        )
    elif hyperparams['optimizer_type'] == 'SGD':
        optimizer = SGD(
            learning_rate=hyperparams.get('learning_rate', 1e-6),
            momentum=hyperparams.get('momentum', 0.0),
            decay=hyperparams.get('decay', 0.0),
            nesterov=True
        )
    elif hyperparams['optimizer_type'] == 'RMSprop':
        optimizer = RMSprop(
            learning_rate=hyperparams.get('learning_rate', 1e-6),
            rho=hyperparams.get('rho', 0.9),
            decay=hyperparams.get('decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {hyperparams['optimizer_type']}")

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

    return model
