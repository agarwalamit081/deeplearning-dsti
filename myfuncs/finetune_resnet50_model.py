# finetune_resnet50_model.py
# Finetune ResNet50 model

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
from myfuncs.custom_layer import CustomLayer
from myfuncs.metrics import metrics

def finetune_resnet50_model(base_model, transfer_layer, num_classes, hyperparams):
    """
    Fine-tune the ResNet50 model with dynamic hyperparameters.

    Args:
    - base_model: The pre-trained ResNet50 base model.
    - transfer_layer: The layer where transfer learning should start.
    - hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).

    Returns:
    - finetune_model: The final Keras model ready for training.
    """
    
    # Extract necessary hyperparameters dynamically
    dropout = hyperparams.get('dropout', 0.5)
    fc_layers = hyperparams.get('fc_layers', [512])
    x_trainable = hyperparams.get('x_trainable', 'all')
    new_weights = hyperparams.get('new_weights', "")

    # Freeze layers conditionally
    freezed_layers = 0
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
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers += 1

    all_layers = len(base_model.layers)
    print(f"Number of all layers in the feature-extractor part of the model: {all_layers}.")
    print(f"Number of frozen (untrainable) layers in the feature-extractor part of the model: {freezed_layers}.")

    # Adding custom layers to the classification part of the model
    x = transfer_layer.output
    x = SpatialDropout2D(0.2)(x)                # Adding Spatial Dropout to convolutional layers
    x = AttentionBlock()(x)                     # Adding attention mechanism
    x = SEBlock()(x)                            # Adding SE-Block
    x = CustomLayer(block_type='attention')(x)  # Using CustomLayer as AttentionBlock
    x = CustomLayer(block_type='se')(x)         # Using CustomLayer as SEBlock

    # Add fully connected layers dynamically
    x = GlobalAveragePooling2D()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)  

    # Create the final model
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    # Optionally load new weights if provided
    if new_weights != "":
        finetune_model.load_weights(new_weights)

    return finetune_model
