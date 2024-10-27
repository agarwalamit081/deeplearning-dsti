# objective.py
# Define the objective function

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from myfuncs.create_model_for_optuna import create_model_for_optuna

def objective(params, train_gen):
    '''
    Objective function to reduce evaluation loss
    '''
    model = KerasClassifier(build_fn=lambda: create_model_for_optuna(params), epochs=5, batch_size=4, verbose=0)

    # Collect a sample of data from the generator
    X_sample, y_sample = [], []
    for i in range(2):  # Gather a few batches
        X, y = train_gen[i]
        X_sample.append(X)
        y_sample.append(y)

    X_sample = np.concatenate(X_sample, axis=0)
    y_sample = np.concatenate(y_sample, axis=0)

    # Handle NaNs and infinity values in the dataset
    y_sample = np.nan_to_num(y_sample, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isinf(y_sample).any():
        raise ValueError("y_sample contains infinity values.")

    X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
#     model.compile(optimizer=model.optimizer, loss='categorical_crossentropy', metrics=metrics)
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=4,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    val_loss = history.history['val_loss'][-1]
    return val_loss
