# create_callbacks.py
# Callbacks for early stopping, learning rate, checkpoint model

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

def create_callbacks(model_name, fold_no):
    '''
    Callbacks for early stopping, learning rate, checkpoint model
    '''


    monit_acc = 'val_top_3_acc'  # For monitoring accuracy
    monit_loss = 'val_loss'      # For monitoring loss
    mode = 'max'                 # For accuracy-based monitoring
    loss_mode = 'min'            # For loss-based monitoring

    callbacks = [
        # Early stopping will monitor the validation loss instead of accuracy for stopping criteria
        EarlyStopping(monitor=monit_loss, mode=loss_mode, verbose=1, patience=5, restore_best_weights=True),
        # EarlyStopping(monitor=monit, mode=mode, verbose=1, patience=5, restore_best_weights=True),

        # Reduce learning rate based on validation loss (to help with better convergence)
        ReduceLROnPlateau(monitor=monit_loss, mode=loss_mode, verbose=1, patience=2, factor=0.2, min_lr=1e-6),
        # ReduceLROnPlateau(monitor=monit, mode=mode, verbose=1, patience=2, factor=0.2, min_lr=1e-6),

        # Save the model only when the top 3 accuracy improves
        ModelCheckpoint(f'{model_name}_fold_{fold_no}.h5', monitor=monit_acc, mode=mode, verbose=1, save_best_only=True)
    ]
    
    return callbacks

