# finetune_efficientnet_model.py
# Finetune EfficientNet model

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Multiply, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from myfuncs.se_block import SEBlock
from myfuncs.attention_block import AttentionBlock
from myfuncs.custom_layer import CustomLayer
from myfuncs.metrics import metrics

def finetune_efficientnet_model(base_model, num_classes, learning_rate=1e-6, weight_decay=1e-6, x_trainable='all'):
    # Fine-tuning EfficientNet with additional layers
    x = base_model.output
    x = SpatialDropout2D(0.2)(x)  # Adding Spatial Dropout to convolutional layers

    x = AttentionBlock()(x)  # Adding attention mechanism
    x = SEBlock()(x)  # Adding SE-Block
    x = CustomLayer(block_type='attention')(x)  # Using CustomLayer as AttentionBlock
    x = CustomLayer(block_type='se')(x)         # Using CustomLayer as SEBlock
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=weight_decay)
    finetune_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    return finetune_model

