# plot_test_results.py
# Plot the test results

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
import matplotlib.pyplot as plt

def plot_test_results(history, model_name=None):
    '''
    Function to plot the test results
    '''

    models = list(test_results.keys())
    test_accuracy = [test_results[model]['accuracy'] for model in models]
    test_loss = [test_results[model]['loss'] for model in models]
    top_3_acc = [test_results[model]['top_3_acc'] for model in models]
    top_5_acc = [test_results[model]['top_5_acc'] for model in models]

    plt.figure(figsize=(14, 8))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.barh(models, test_accuracy, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Test Accuracy for Different Models')

    # Plot loss
    plt.subplot(2, 2, 2)
    plt.barh(models, test_loss, color='salmon')
    plt.xlabel('Loss')
    plt.title('Test Loss for Different Models')

    # Plot top 3 accuracy
    plt.subplot(2, 2, 3)
    plt.barh(models, top_3_acc, color='lightgreen')
    plt.xlabel('Top 3 Accuracy')
    plt.title('Test Top 3 Accuracy for Different Models')

    # Plot top 5 accuracy
    plt.subplot(2, 2, 4)
    plt.barh(models, top_5_acc, color='lightcoral')
    plt.xlabel('Top 5 Accuracy')
    plt.title('Test Top 5 Accuracy for Different Models')

    plt.tight_layout()
    plt.show()
