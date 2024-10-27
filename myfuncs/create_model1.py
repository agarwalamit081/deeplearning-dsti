# create_model1.py
# Model 1

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from myfuncs.metrics import metrics

def create_model1(input_shape, num_classes, hyperparams):
    """
    Create Model 1 with dynamic hyperparameters.

    Args:
    - input_shape: Shape of the input data.
    - num_classes: Number of output classes.
    - hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).

    Returns:
    - model: Compiled Keras model.
    """
    
    # Extract hyperparameters from the dictionary, using defaults if not provided
    learning_rate = hyperparams.get('learning_rate', 1e-6)
    weight_decay = hyperparams.get('weight_decay', 1e-6)
    dropout = hyperparams.get('dropout', 0.5)
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')

    # Define the model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(weight_decay)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Keep this dropout static since it’s internal to the architecture
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dropout(dropout),  # Use the dynamically provided dropout for the final layers
        Dense(num_classes, activation='softmax')
    ])

    # Select the optimizer based on hyperparams['optimizer_type']
    if optimizer_type == 'Adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=hyperparams.get('beta_1', 0.9),
            beta_2=hyperparams.get('beta_2', 0.999),
            decay=hyperparams.get('decay', 0.0)
        )
    elif optimizer_type == 'SGD':
        optimizer = SGD(
            learning_rate=learning_rate,
            momentum=hyperparams.get('momentum', 0.0),
            decay=hyperparams.get('decay', 0.0),
            nesterov=True
        )
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(
            learning_rate=learning_rate,
            rho=hyperparams.get('rho', 0.9),
            decay=hyperparams.get('decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

    return model
