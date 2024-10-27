# finetune_alexnet_model.py

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, GlobalAveragePooling2D, Multiply, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input
from myfuncs.se_block import SEBlock
from myfuncs.attention_block import AttentionBlock
from myfuncs.custom_layer import CustomLayer
from myfuncs.metrics import metrics

def finetune_alexnet_model(input_shape, num_classes, hyperparams, x_trainable='all'):
    """
    Finetune AlexNet model using dynamic hyperparameters.
    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes for the output layer.
    :param hyperparams: Dictionary of hyperparameters (best_params or default_hyperparams).
    :param x_trainable: Specifies whether to train all layers or freeze some.
    """
    
    # Extract hyperparameters dynamically
    learning_rate = hyperparams.get('learning_rate', 1e-6)
    weight_decay = hyperparams.get('weight_decay', 1e-6)
    dropout = hyperparams.get('dropout', 0.5)
    beta_1 = hyperparams.get('beta_1', 0.9)
    beta_2 = hyperparams.get('beta_2', 0.999)
    decay = hyperparams.get('decay', 0.0)

    input_layer = Input(shape=input_shape)

    # AlexNet layers
    x = Conv2D(filters=96, kernel_size=(11, 11), padding='valid', activation='relu', strides=(4, 4))(input_layer)
    x = SpatialDropout2D(0.2)(x)  # Adding Spatial Dropout to convolutional layers
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = SpatialDropout2D(0.2)(x)  # Adding Spatial Dropout to convolutional layers
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    # Adding custom attention and SE blocks
    x = AttentionBlock()(x)                     # Adding Attention Mechanism
    x = SEBlock()(x)                            # Adding SE-Block
    x = CustomLayer(block_type='attention')(x)  # Using CustomLayer as AttentionBlock
    x = CustomLayer(block_type='se')(x)         # Using CustomLayer as SEBlock

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(dropout)(x)  # Using dynamic dropout value from hyperparams
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(dropout)(x)  # Using dynamic dropout value from hyperparams

    output_layer = Dense(units=num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=input_layer, outputs=output_layer)

    # Select optimizer dynamically based on hyperparameters
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')  # Default is 'Adam' if not provided

    if optimizer_type == 'Adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            decay=decay
        )
    elif optimizer_type == 'SGD':
        optimizer = SGD(
            learning_rate=learning_rate,
            momentum=hyperparams.get('momentum', 0.9),
            decay=decay,
            nesterov=True
        )
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(
            learning_rate=learning_rate,
            rho=hyperparams.get('rho', 0.9),
            decay=decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Compile the model
    finetune_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

    return finetune_model


