# finetune_vgg16_model.py
# Finetune VGG16 model

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Multiply, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model
from myfuncs.se_block import SEBlock
from myfuncs.attention_block import AttentionBlock
from myfuncs.custom_layer import CustomLayer
from myfuncs.metrics import metrics

def finetune_vgg16_model(base_model, transfer_layer, x_trainable, dropout, fc_layers, num_classes, new_weights=""):
    # Freeze layers if required
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
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freezed_layers))
    
    # Adding custom layers to the classification part of the model
    x = transfer_layer.output
    x = SpatialDropout2D(0.2)(x)  # Adding Spatial Dropout to convolutional layers
    x = AttentionBlock()(x)       # Adding attention mechanism
    x = SEBlock()(x)              # Adding SE-Block
    x = CustomLayer(block_type='attention')(x)  # Using CustomLayer as AttentionBlock
    x = CustomLayer(block_type='se')(x)         # Using CustomLayer as SEBlock
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights != "":
        finetune_model.load_weights(new_weights)
    return finetune_model
