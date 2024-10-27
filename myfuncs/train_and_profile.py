# train_and_profile.py
# Train and profile

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

import os
import pickle

import cProfile
import pstats

from myfuncs.train_model import train_model
from myfuncs.extract_history_data import extract_history_data
from myfuncs.gradient_accumulation import GradientAccumulation

def train_and_profile(model_name, kf, train_file_paths, train_labels, batch_size, augmentations, size, metrics, epochs, all_histories):
    def run_training():
        history = train_model(
            model_name=model_name,
            kf=kf,
            train_file_paths=train_file_paths,
            train_labels=train_labels,
            batch_size=batch_size,
            augmentations=augmentations,
            size=size,
            metrics=metrics,
            epochs=epochs
        )
        return history

    # Run the training with profiling
    profile = cProfile.Profile()
    profile.enable()
    history = run_training()
    profile.disable()

    # Save profiling results
    profile_file = f'prof_res_train_model_{model_name}.prof'
    profile.dump_stats(profile_file)

    # Process history
    if history:
        all_histories.append((model_name, history))  # Append to the provided all_histories list
        # Save history
        with open(f'{model_name}_histories.pkl', 'wb') as f:
            pickle.dump(all_histories, f)
        
        # Save the profiling results in a more human-readable form
        with open(f'prof_out_{model_name}.txt', 'w') as f:
            stats = pstats.Stats(profile, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats()
    else:
        print("No history was returned from the training process.")

    return all_histories


## Works in case of fallback but all_history is still None - do not delete
# def train_and_profile(model_name, kf, train_file_paths, train_labels, batch_size, augmentations, size, metrics, epochs):
#     all_histories = []

#     def run_training():
#         history = train_model(
#             model_name=model_name,
#             kf=kf,
#             train_file_paths=train_file_paths,
#             train_labels=train_labels,
#             batch_size=batch_size,
#             augmentations=augmentations,
#             size=size,
#             metrics=metrics,
#             epochs=epochs
#         )
#         return history

#     # Run the training with profiling
#     profile = cProfile.Profile()
#     profile.enable()
#     history = run_training()
#     profile.disable()

#     # Save profiling results
#     profile_file = f'prof_res_train_model_{model_name}.prof'
#     profile.dump_stats(profile_file)

#     # Process history
#     if history:
#         all_histories.append((model_name, history))
#         # Save history
#         with open(f'{model_name}_histories.pkl', 'wb') as f:
#             pickle.dump(all_histories, f)
        
#         # Save the profiling results in a more human-readable form
#         with open(f'prof_out_{model_name}.txt', 'w') as f:
#             stats = pstats.Stats(profile, stream=f)
#             stats.sort_stats('cumulative')
#             stats.print_stats()
#     else:
#         print("No history was returned from the training process.")

#     return all_histories
