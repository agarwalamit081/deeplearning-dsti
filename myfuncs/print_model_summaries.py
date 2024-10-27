# print_model_summaries.py

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
import tensorflow as tf

def print_model_summaries(models, model_names):
    """
    Function to print and save model summaries.
    
    Args:
    - models: List of model functions.
    - model_names: List of model names.
    """
    for model, name in zip(models, model_names):
        print(f"Summary for {name}:")
        model.summary()  # Print the summary of the model architecture

        # Optionally, save the summary to a text file
        with open(f'{name}_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))  # Save summary to a text file



