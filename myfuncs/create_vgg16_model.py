# create_vgg16_model.py
# Model 3 - VGG16

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from myfuncs.finetune_vgg16_model import finetune_vgg16_model
from myfuncs.metrics import metrics

def create_vgg16_model(input_shape, num_classes, hyperparams):
    """
    Create and fine-tune the VGG16 model using dynamic hyperparameters.

    Args:
    - input_shape: Shape of the input data.
    - num_classes: Number of output classes.
    - hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).

    Returns:
    - model: Compiled VGG16 model.
    """
    
    # Extract necessary hyperparameters dynamically from the dictionary
    learning_rate = hyperparams.get('learning_rate', 1e-6)
    weight_decay = hyperparams.get('weight_decay', 1e-6)
    dropout = hyperparams.get('dropout', 0.5)
    fc_layers = hyperparams.get('fc_layers', [512])  # Default fully connected layer size
    x_trainable = hyperparams.get('x_trainable', 'all')
    new_weights = hyperparams.get('new_weights', "")
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')  # Default to 'Adam'

    # Load the pre-trained VGG16 model without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    transfer_layer = base_model.get_layer('block5_pool')

    # Fine-tune the model using the finetune_vgg16_model function
    model = finetune_vgg16_model(
        base_model=base_model, 
        transfer_layer=transfer_layer, 
        x_trainable=x_trainable, 
        dropout=dropout, 
        fc_layers=fc_layers, 
        num_classes=num_classes, 
        new_weights=new_weights
    )

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

    # Compile the model with the selected optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    
    return model
