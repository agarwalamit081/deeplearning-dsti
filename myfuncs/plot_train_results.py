# plot_train_results.py
# Plot the final train accuracy and loss

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
import matplotlib.pyplot as plt

def plot_train_results(history_dict, model_name=None):
    '''
    Function to plot the training results (accuracy, loss, etc.)
    '''
    plt.figure(figsize=(16, 12))

    # Plot training & validation accuracy values
    plt.subplot(3, 3, 1)
    plt.plot(history_dict['accuracy'], label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy {model_name}' if model_name else 'Training and Validation Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(3, 3, 2)
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss {model_name}' if model_name else 'Training and Validation Loss')
    plt.legend()

    # Plot training & validation Top 3 Accuracy values
    plt.subplot(3, 3, 3)
    plt.plot(history_dict['top_3_acc'], label='Train Top 3 Accuracy')
    plt.plot(history_dict['val_top_3_acc'], label='Validation Top 3 Accuracy')
    plt.title(f'Training and Validation Top 3 Accuracy {model_name}' if model_name else 'Training and Validation Top 3 Accuracy')
    plt.legend()

    # Plot training & validation Top 5 Accuracy values
    plt.subplot(3, 3, 4)
    plt.plot(history_dict['top_5_acc'], label='Train Top 5 Accuracy')
    plt.plot(history_dict['val_top_5_acc'], label='Validation Top 5 Accuracy')
    plt.title(f'Training and Validation Top 5 Accuracy {model_name}' if model_name else 'Training and Validation Top 5 Accuracy')
    plt.legend()

    # Plot training & validation Precision values
    plt.subplot(3, 3, 5)
    plt.plot(history_dict['precision'], label='Train Precision')
    plt.plot(history_dict['val_precision'], label='Validation Precision')
    plt.title(f'Training and Validation Precision {model_name}' if model_name else 'Training and Validation Precision')
    plt.legend()

    # Plot training & validation Recall values
    plt.subplot(3, 3, 6)
    plt.plot(history_dict['recall'], label='Train Recall')
    plt.plot(history_dict['val_recall'], label='Validation Recall')
    plt.title(f'Training and Validation Recall {model_name}' if model_name else 'Training and Validation Recall')
    plt.legend()

    # Plot training & validation AUC values
    plt.subplot(3, 3, 7)
    plt.plot(history_dict['auc'], label='Train AUC')
    plt.plot(history_dict['val_auc'], label='Validation AUC')
    plt.title(f'Training and Validation AUC {model_name}' if model_name else 'Training and Validation AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()
