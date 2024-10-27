# finetune_imagenet_model.py
# Finetune ImageNet model

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Multiply, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from myfuncs.se_block import SEBlock
from myfuncs.attention_block import AttentionBlock
from myfuncs.custom_layer import CustomLayer
from myfuncs.metrics import metrics

def finetune_imagenet_model(input_shape, num_classes, hyperparams):
    """
    Fine-tune the ImageNet model with dynamic hyperparameters.
    
    Args:
    - input_shape: Input shape of the model.
    - num_classes: Number of output classes.
    - hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).
    """
    
    # Extract necessary hyperparameters dynamically
    learning_rate = hyperparams.get('learning_rate', 1e-6)
    weight_decay = hyperparams.get('weight_decay', 1e-6)
    dropout = hyperparams.get('dropout', 0.5)
    fc_layers = hyperparams.get('fc_layers', [512])
    x_trainable = hyperparams.get('x_trainable', 'all')

    # Load the pre-trained ResNet50 model without the top classification layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freezing layers conditionally
    if x_trainable == 'partial':
        for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4 layers
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True if x_trainable == 'all' else False

    # Model architecture
    x = base_model.output
    x = SpatialDropout2D(0.2)(x)  # Adding Spatial Dropout to convolutional layers

    # Using CustomLayer and dedicated SE/Attention Blocks
    x = AttentionBlock()(x)  # Adding attention mechanism
    x = SEBlock()(x)  # Adding SE-Block
    x = CustomLayer(block_type='attention')(x)
    x = CustomLayer(block_type='se')(x)
    
    # Add fully connected layers dynamically
    x = Flatten()(x)
    for units in fc_layers:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the full model
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    # Select the optimizer based on hyperparams['optimizer_type']
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
    finetune_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    
    return finetune_model
