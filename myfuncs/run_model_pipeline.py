# run_model_pipeline.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
import inspect
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from myfuncs.gradient_accumulation import GradientAccumulation
from myfuncs.create_callbacks import create_callbacks
from myfuncs.plot_train_results import plot_train_results
from myfuncs.metrics import metrics


def run_model_pipeline(model_func, model_name, hyperparams, input_shape, num_classes, train_generator, test_generator, epochs, accum_steps=4):
    # Get the signature of the model function
    model_func_signature = inspect.signature(model_func)
    
    # Filter the hyperparameters based on the model function's arguments
    filtered_hyperparams = {k: v for k, v in hyperparams.items() if k in model_func_signature.parameters}

    # Handle specific cases like DenseNet, where dropout is not needed
    if model_name == 'DenseNet':
        if 'dropout' in filtered_hyperparams:
            del filtered_hyperparams['dropout']

    # Create the model with filtered hyperparameters
    model = model_func(
        input_shape=input_shape,
        num_classes=num_classes,
        hyperparams=hyperparams   # Pass the entire hyperparams dictionary
        # **filtered_hyperparams  # Pass only relevant hyperparameters
    )

    # Print model summary to compare model architectures
    print(f"Summary for {model_name}:")
    model.summary()  # This will print the summary of the model architecture
    
    # Save the model summary to a file (optional)
    with open(f'{model_name}_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))  # Save model summary to a text file

    # Select the optimizer dynamically based on hyperparams['optimizer_type']
    optimizer_type = hyperparams.get('optimizer_type', 'Adam')  # Default is 'Adam' if not provided

    if optimizer_type == 'Adam':
        optimizer = Adam(
            learning_rate=hyperparams.get('learning_rate', 1e-3),
            beta_1=hyperparams.get('beta_1', 0.9),
            beta_2=hyperparams.get('beta_2', 0.999),
            decay=hyperparams.get('decay', 0.0)
        )
    elif optimizer_type == 'SGD':
        optimizer = SGD(
            learning_rate=hyperparams.get('learning_rate', 1e-3),
            momentum=hyperparams.get('momentum', 0.0),
            decay=hyperparams.get('decay', 0.0),
            nesterov=True
        )
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(
            learning_rate=hyperparams.get('learning_rate', 1e-3),
            rho=hyperparams.get('rho', 0.9),
            decay=hyperparams.get('decay', 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Compile the model with the selected optimizer
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
    )

    # Define Callbacks
    fold_no = 1
    callbacks = create_callbacks(model_name=model_name, fold_no=fold_no)

    # Add Gradient Accumulation callback
    callbacks.append(GradientAccumulation(accum_steps=accum_steps))

    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=callbacks)

    # Save the model and history
    model.save(f'{model_name}_trained.h5')

    # Plot the training results
    plot_train_results(history.history)

    return history.history
