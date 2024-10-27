# inspect_histories.py
# Inspect the keys in the histories to ensure they were not NULL

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

def inspect_histories(all_histories):
    '''
    # Function to inspect the keys in the histories
    '''
    for model_name, histories in all_histories:
        print(f"Model: {model_name}")
        for i, history in enumerate(histories):
            print(f"  Fold {i + 1} keys: {list(history.keys())}")
