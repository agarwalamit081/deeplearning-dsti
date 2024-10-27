# finetune_inceptionv3.py
# Finetune InceptionV3 model

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Multiply, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model
from myfuncs.se_block import SEBlock
from myfuncs.attention_block import AttentionBlock
from myfuncs.metrics import metrics

def finetune_inceptionv3(base_model, transfer_layer, x_trainable='all', dropout=0.5, fc_layers=[512], num_classes=1000, new_weights=""):
    """
    Fine-tune the InceptionV3 model with dynamic hyperparameters.
    
    Args:
    - base_model: The pre-trained InceptionV3 base model.
    - transfer_layer: Layer from the base model to start fine-tuning from.
    - x_trainable: Specifies which layers are trainable ('all', 'partial', or an integer for custom).
    - dropout: Dropout rate for regularization.
    - fc_layers: List specifying the size of fully connected layers.
    - num_classes: Number of output classes.
    - new_weights: Optional path to load specific pre-trained weights.
    
    Returns:
    - finetune_model: The final fine-tuned model ready for training.
    """
    freezed_layers = 0
    all_layers = len(base_model.layers)
    
    # Conditionally freeze layers based on 'x_trainable'
    if x_trainable != "all":
        if x_trainable == "none":
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers += 1
        elif x_trainable == "partial":
            for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4
                layer.trainable = False
                freezed_layers += 1
        else:
            for layer in base_model.layers[:-x_trainable]:  # Freeze all layers except the last 'x_trainable' layers
                layer.trainable = False
                freezed_layers += 1

    print(f"Number of all layers in the feature-extractor part of the model: {all_layers}.")
    print(f"Number of frozen (untrainable) layers in the feature-extractor part of the model: {freezed_layers}.")

    # Add custom layers to the classification part of the model
    x = transfer_layer.output
    x = SpatialDropout2D(0.2)(x)  # Adding Spatial Dropout to convolutional layers
    x = AttentionBlock()(x)  # Adding attention mechanism
    x = SEBlock()(x)  # Adding SE-Block
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout)(x)

    # Add fully connected layers dynamically based on 'fc_layers'
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final fine-tuned model
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    # Optionally load specific pre-trained weights
    if new_weights != "":
        finetune_model.load_weights(new_weights)

    return finetune_model
