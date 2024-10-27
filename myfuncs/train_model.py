# train_model.py
# Train a specific model

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

import os
import pickle

from myfuncs.create_model import create_model
from myfuncs.create_callbacks import create_callbacks
from myfuncs.load_hyperparameters import load_hyperparameters
from myfuncs.perform_hyperparameter_search import perform_hyperparameter_search
from myfuncs.extract_history_data import extract_history_data
from myfuncs.save_hyperparameters import save_hyperparameters
from myfuncs.gradient_accumulation import GradientAccumulation
from myfuncs.get_image_paths_and_labels import get_image_paths_and_labels
from myfuncs.AlbumentationsImageDataGenerator import AlbumentationsImageDataGenerator

def train_model(model_name, kf, train_file_paths, train_labels, batch_size, augmentations, size, metrics, epochs):
    '''
    Train a specific model using the best parameter found previously and using data augmentation 
    and callbacks for early stopping, model checkpointing, and reducing the learning rate on the plateau.
    '''
    histories = []
    fold_no = 1  # Reset fold number for each model

    for train_index, val_index in kf.split(train_file_paths):
        print(f'Training fold {fold_no} for model {model_name}...')
        train_paths, val_paths = np.array(train_file_paths)[train_index], np.array(train_file_paths)[val_index]
        train_labels_fold, val_labels_fold = train_labels[train_index], train_labels[val_index]
        
        train_gen_fold = AlbumentationsImageDataGenerator(train_paths, train_labels_fold, batch_size, augmentations, size)
        test_gen = AlbumentationsImageDataGenerator(val_paths, val_labels_fold, batch_size, None, size)

        # Load or perform hyperparameter search
        best_params = load_hyperparameters(model_name)
        if best_params is None:
            print("Performing hyperparameter search...")
            best_params = perform_hyperparameter_search(train_gen_fold)
            print("Best parameters found:", best_params)
            save_hyperparameters(model_name, best_params)
        else:
            print("Loaded hyperparameters from file:", best_params)

        model = create_model(input_shape=(size[0], size[1], 3), num_classes=train_labels.shape[1], \
                             optimizer_type=best_params['optimizer_type'], \
                             learning_rate=best_params['learning_rate'], \
                             weight_decay=best_params['weight_decay'], \
                             dropout=best_params['dropout'], beta_1=best_params['beta_1'], \
                             beta_2=best_params['beta_2'], decay=best_params['decay'], \
                             momentum=best_params['momentum'], rho=best_params['rho'])
        
        model.compile(optimizer=model.optimizer, loss='categorical_crossentropy', metrics=metrics)

        callbacks = create_callbacks(model_name, fold_no)
        callbacks.append(GradientAccumulation(accum_steps=4))

        history = model.fit(train_gen_fold, epochs=epochs, validation_data=test_gen, callbacks=callbacks, verbose=1)
        histories.append(history)

        fold_no += 1

    # Extract and save the histories for the model using pickle
    extracted_histories = extract_history_data(histories)
    with open(f'{model_name}_histories.pkl', 'wb') as f:
        pickle.dump(extracted_histories, f)

    print("Histories captured for model:", histories)
    return extracted_histories
