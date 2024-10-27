# get_test_data.py
# load test data and labels

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

from myfuncs.get_image_paths_and_labels import get_image_paths_and_labels

def get_test_data(test_anno, test_dir, names_df):
    '''
    Function to load test data and labels
    '''
    test_file_paths, test_labels = get_image_paths_and_labels(test_anno, test_dir, names_df)
    test_labels = pd.get_dummies(test_labels).values
    return test_file_paths, test_labels
