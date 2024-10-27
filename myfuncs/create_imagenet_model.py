# create_imagenet_model.py
# Model 6 - ImageNet

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from myfuncs.finetune_imagenet_model import finetune_imagenet_model
from myfuncs.metrics import metrics

def create_imagenet_model(input_shape, num_classes, hyperparams):
    """
    Create ImageNet model with dynamic hyperparameters.

    Args:
    - input_shape: The input shape for the model.
    - num_classes: The number of classes for classification.
    - hyperparams: Dictionary of hyperparameters (default_hyperparams or best_params).

    Returns:
    - model: A compiled model ready for training.
    """

    # Load the ResNet50 base model without top classification layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Extract the x_trainable flag from hyperparams
    x_trainable = hyperparams.get('x_trainable', 'all')

    # Freezing layers conditionally
    if x_trainable == 'partial':
        for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4 layers
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True if x_trainable == 'all' else False

    # Fine-tune the model using the hyperparameters
    model = finetune_imagenet_model(input_shape=input_shape, num_classes=num_classes, hyperparams=hyperparams)

    return model
