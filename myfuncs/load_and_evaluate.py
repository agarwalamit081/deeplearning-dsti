# load_and_evaluate.py
# Define the data loading and evaluation process

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
import pickle

from myfuncs.evaluate_model_on_test_data import evaluate_model_on_test_data

def load_and_evaluate(test_file_paths, test_labels, size, batch_size):
    '''
    Function to define the data loading and evaluation process
    '''
    test_gen = AlbumentationsImageDataGenerator(test_file_paths, test_labels, batch_size, None, size)
    for model_name, _ in model_creators:
        print(f'Evaluating {model_name} on the test data...')
        # best_model = load_model(f'{model_name}.h5')
        best_model = load_model(f'{model_name}_fold_{fold_no-1}.h5')
        loss, accuracy, top_3_acc, top_5_acc = best_model.evaluate(test_gen)
        print(f'{model_name} Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}, \
        Top-3 Accuracy: {top_3_acc:.4f}, Top-5 Accuracy: {top_5_acc:.4f}')
        return loss, accuracy, top_3_acc, top_5_acc
