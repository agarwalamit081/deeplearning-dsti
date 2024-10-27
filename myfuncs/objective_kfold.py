# objective_kfold.py
# Define the objective function
# Uses k-fold which makes it run out of memory

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from myfuncs.create_model_for_optuna import create_model_for_optuna

def objective(params, train_gen, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_val_losses = []

    for train_index, val_index in kf.split(train_gen):
        train_gen_fold = [train_gen[i] for i in train_index]
        val_gen_fold = [train_gen[i] for i in val_index]

        X_train, y_train = [], []
        for X, y in train_gen_fold:
            X_train.append(X)
            y_train.append(y)

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        X_val, y_val = [], []
        for X, y in val_gen_fold:
            X_val.append(X)
            y_val.append(y)

        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        model = create_model_for_optuna(params)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        val_loss = history.history['val_loss'][-1]
        all_val_losses.append(val_loss)

    return np.mean(all_val_losses)
