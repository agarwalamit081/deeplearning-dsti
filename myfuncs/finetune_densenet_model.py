# finetune_densenet_model.py
# Finetune DenseNet model

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Multiply, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from myfuncs.metrics import metrics
from myfuncs.attention_block import AttentionBlock
from myfuncs.se_block import SEBlock

def finetune_densenet_model(base_model, num_classes, hyperparams):
    """
    Function to fine-tune the DenseNet model.
    Args:
    - base_model: The pre-trained DenseNet base model.
    - num_classes: Number of output classes.
    - hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).
    """

    # Extract hyperparameters dynamically
    fc_layers = hyperparams.get('fc_layers', [512])  # Default fully connected layer size
    dropout = hyperparams.get('dropout', 0.5)        # Dropout rate
    spatial_dropout_rate = hyperparams.get('spatial_dropout', 0.2)  # Spatial dropout rate

    x = base_model.output
    x = SpatialDropout2D(spatial_dropout_rate)(x)  # Adding Spatial Dropout to convolutional layers

    # Attention and SE blocks
    x = AttentionBlock()(x)  # Adding attention mechanism
    x = SEBlock()(x)  # Adding SE-Block

    x = Flatten()(x)

    # Add fully connected layers dynamically from fc_layers
    for units in fc_layers:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)

    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create final fine-tuned model
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model
