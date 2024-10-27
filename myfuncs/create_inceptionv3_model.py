# create_inceptionv3_model.py
# Model 8 - inceptionv3

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from myfuncs.finetune_inceptionv3 import finetune_inceptionv3
from myfuncs.metrics import metrics

def create_inceptionv3_model(input_shape, num_classes, hyperparams):
    """
    Create and compile the InceptionV3 model with dynamic hyperparameters.
    
    Args:
    - input_shape: Input shape of the model.
    - num_classes: Number of output classes.
    - hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).
    
    Returns:
    - model: Compiled InceptionV3 model.
    """
    
    # Extract necessary hyperparameters dynamically
    learning_rate = hyperparams.get('learning_rate', 1e-4)
    weight_decay = hyperparams.get('weight_decay', 1e-4)
    dropout = hyperparams.get('dropout', 0.5)
    fc_layers = hyperparams.get('fc_layers', [512])
    x_trainable = hyperparams.get('x_trainable', 'all')
    new_weights = hyperparams.get('new_weights', "")
    
    # Load the pre-trained InceptionV3 model without the top classification layers
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    transfer_layer = base_model.get_layer('mixed10')  # Extract the transfer layer

    # Fine-tune the model using the finetune_inceptionv3 function
    model = finetune_inceptionv3(
        base_model=base_model, 
        transfer_layer=transfer_layer, 
        x_trainable=x_trainable, 
        dropout=dropout, 
        fc_layers=fc_layers, 
        num_classes=num_classes, 
        new_weights=new_weights
    )

    # Select the optimizer based on hyperparams['optimizer_type']
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')  # Set a default if not provided
    if hyperparams['optimizer_type'] == 'Adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=hyperparams.get('beta_1', 0.9),
            beta_2=hyperparams.get('beta_2', 0.999),
            decay=hyperparams.get('decay', 0.0)
        )
    elif hyperparams['optimizer_type'] == 'SGD':
        optimizer = SGD(
            learning_rate=learning_rate,
            momentum=hyperparams.get('momentum', 0.0),
            decay=hyperparams.get('decay', 0.0),
            nesterov=True
        )
    elif hyperparams['optimizer_type'] == 'RMSprop':
        optimizer = RMSprop(
            learning_rate=learning_rate,
            rho=hyperparams.get('rho', 0.9),
            decay=hyperparams.get('decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {hyperparams['optimizer_type']}")

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    
    return model
