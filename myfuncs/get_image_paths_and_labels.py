# get_image_paths_and_labels.py
# Load image paths and labels

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd
import os

def get_image_paths_and_labels(data, directory, names_df):
    '''
    Function to load image paths and labels
    '''
    paths = []
    labels = []
    for index, row in data.iterrows():
        fname = row['fname']
        class_id = row['class']
        class_name = names_df.iloc[class_id - 1, 0]  # Keep the original class name with spaces
        class_name = class_name.replace('_', ' ').replace('\\', ' ').replace('/', ' ')
        if "Ram C V" in class_name:
            class_name = class_name.replace("Ram C V", "Ram C-V")
        sub_dir = os.path.join(directory, class_name).replace('\\', '/')
        file_path = os.path.join(sub_dir, fname).replace('\\', '/')
        if os.path.exists(file_path):
            paths.append(file_path)
            labels.append(class_id)
        else:
            print(f"File not found: {file_path}")
    return paths, labels
