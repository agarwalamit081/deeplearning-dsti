# mixup.py
# Mixup

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

import tensorflow as tf

def mixup(image, label, alpha=0.2):
    '''
    An augmentation technique where images and their corresponding labels are mixed.
    '''

    """Apply Mixup to the batch."""
    batch_size = len(images)
    indices = np.random.permutation(batch_size)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    lam = np.random.beta(alpha, alpha)
    images = lam * images + (1 - lam) * shuffled_images
    labels = lam * labels + (1 - lam) * shuffled_labels

    # return images, labels

    # Sample lambda from the beta distribution
    beta = tf.random.uniform([], 0, alpha)
    
    # Get indices for the mixup
    index = tf.random.shuffle(tf.range(tf.shape(image)[0]))
    
    # Get the images and labels
    image_a, image_b = image, tf.gather(image, index)
    label_a, label_b = label, tf.gather(label, index)
    
    new_image = image_a * beta + image_b * (1. - beta)
    new_label = label_a * beta + label_b * (1. - beta)
    
    return new_image, new_label
