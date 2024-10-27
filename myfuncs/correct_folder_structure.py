# correct_folder_structure.py

import os
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

def correct_folder_structure(augmented_dir, car_models):
    '''
        This function corrects the directory structure for augmented images of different car models.
        Specifically, it ensures that each car model has the correct folder name, creates missing folders, and removes unnecessary or incorrectly named directories.
        Input:
            augmented_dir (string): Path to the base directory where the folders for car models will be created or corrected.
            car_models (list or array-like): A list or array of car model names.
    
       Output:
           No return value (None): The function modifies the directory structure on the file system.
    '''
    for car_model in car_models:
        correct_name = car_model.replace("Ram C V", "Ram C-V")
        correct_dir = os.path.join(augmented_dir, correct_name)
        if not os.path.exists(correct_dir):
            os.makedirs(correct_dir)

        # Remove any incorrect directories if they exist
        incorrect_dir = os.path.join(augmented_dir, "Ram C")
        if os.path.exists(incorrect_dir):
            try:
                os.rmdir(incorrect_dir)  # Only removes if empty
            except OSError:
                pass
