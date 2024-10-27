# create_densenet_model.py
# Model 5 - DenseNet

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from myfuncs.finetune_densenet_model import finetune_densenet_model
from myfuncs.metrics import metrics

def create_densenet_model(input_shape, num_classes, hyperparams, x_trainable='all'):
    """
    Create DenseNet model using dynamic hyperparameters.
    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes for the output layer.
    :param hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).
    :param x_trainable: Specifies whether to train all layers or freeze some layers.
    """
    # Load DenseNet121 base model
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    # Set layer trainability based on x_trainable
    if x_trainable == 'partial':
        for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4 layers
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True if x_trainable == 'all' else False

    # Use the updated finetune_densenet_model function
    model = finetune_densenet_model(base_model, num_classes, hyperparams)
    # model = finetune_densenet_model(base_model, num_classes, dropout=dropout, fc_layers=[512])

    # Select optimizer dynamically based on hyperparameters
    learning_rate = hyperparams.get('learning_rate', 1e-6)
    weight_decay = hyperparams.get('weight_decay', 1e-6)
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')
    beta_1 = hyperparams.get('beta_1', 0.9)
    beta_2 = hyperparams.get('beta_2', 0.999)
    decay = hyperparams.get('decay', 0.0)

    # Choose the optimizer based on hyperparameters
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


