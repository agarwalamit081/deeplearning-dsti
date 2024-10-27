# evaluate_model_on_test_data.py
# Evaluate models on test data

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

import os
import pickle

from tensorflow.keras.models import load_model

def evaluate_model_on_test_data(model_name, test_gen, test_results):
    '''
    Function to evaluate models on test data    
    '''
    print(f'Evaluating {model_name} on the test data...')
    best_model_path = f'{model_name}_fold_1.h5'  # assuming the best model is saved as fold_1
    if not os.path.exists(best_model_path):
        print(f"Model file {best_model_path} does not exist.")
        return None
    try:
        best_model = load_model(best_model_path)
        print(f"Loaded model from {best_model_path}")
        results = best_model.evaluate(test_gen)
        metrics = {name: result for name, result in zip(best_model.metrics_names, results)}
        print(f'{model_name} Test results: {metrics}')
        test_results[model_name] = metrics
        return metrics
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None
