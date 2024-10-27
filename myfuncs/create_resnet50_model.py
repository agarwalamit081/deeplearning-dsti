# create_resnet50_model.py
# Model 4 - ResNET50

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from myfuncs.finetune_resnet50_model import finetune_resnet50_model
from myfuncs.metrics import metrics

def create_resnet50_model(input_shape, num_classes, hyperparams):
    """
    Create and fine-tune the ResNet50 model using dynamic hyperparameters.
    
    Args:
    - input_shape: Shape of the input data.
    - num_classes: Number of output classes.
    - hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).
    
    Returns:
    - model: Compiled ResNet50 model.
    """

    # Load the ResNet50 base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    transfer_layer = base_model.get_layer('conv5_block3_out')
    
    # Fine-tune the model using the hyperparameters
    model = finetune_resnet50_model(base_model, transfer_layer, num_classes, hyperparams)

    # Extract optimizer-related hyperparameters from the dictionary
    learning_rate = hyperparams.get('learning_rate', 1e-6)
    weight_decay = hyperparams.get('weight_decay', 1e-6)
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')
    beta_1 = hyperparams.get('beta_1', 0.9)
    beta_2 = hyperparams.get('beta_2', 0.999)
    decay = hyperparams.get('decay', 0.0)

    # Select optimizer dynamically based on hyperparameters
    if optimizer_type == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
    elif optimizer_type == 'SGD':
        optimizer = SGD(learning_rate=learning_rate, momentum=hyperparams.get('momentum', 0.9), decay=decay, nesterov=True)
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate, rho=hyperparams.get('rho', 0.9), decay=decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    
    return model
