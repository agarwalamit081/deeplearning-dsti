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

def finetune_densenet_model(base_model, num_classes, fc_layers=[512], dropout=0.5):
    """
    Function to fine-tune the DenseNet model.
    Args:
    - base_model: The pre-trained DenseNet base model
    - num_classes: Number of output classes
    - fc_layers: List of units for fully connected layers (default: [512])
    - dropout: Dropout rate for fully connected layers (default: 0.5)
    """
    x = base_model.output
    x = SpatialDropout2D(0.2)(x)  # Adding Spatial Dropout to convolutional layers
    x = AttentionBlock()(x)  # Adding attention mechanism
    x = SEBlock()(x)  # Adding SE-Block
    x = Flatten()(x)
    
    # Add fully connected layers dynamically
    for units in fc_layers:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model
