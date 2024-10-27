# cutmix.py
# Cutmix

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool

import pandas as pd

import tensorflow as tf

def cutmix(image, label, PROBABILITY=1.0):
    '''
    CutMix augmentation randomly replaces a part of the image with a patch from another image, 
    and the labels are mixed according to the proportion of the area that was replaced.
    '''

    batch_size = len(images)
    indices = np.random.permutation(batch_size)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    lam = np.random.beta(alpha, alpha)
    rx = np.random.uniform(0, 1)
    ry = np.random.uniform(0, 1)

    r = np.sqrt(1 - lam)
    rw = int(r * images.shape[1])
    rh = int(r * images.shape[2])

    cx = int(rx * images.shape[1])
    cy = int(ry * images.shape[2])

    x1 = np.clip(cx - rw // 2, 0, images.shape[1])
    y1 = np.clip(cy - rh // 2, 0, images.shape[2])
    x2 = np.clip(cx + rw // 2, 0, images.shape[1])
    y2 = np.clip(cy + rh // 2, 0, images.shape[2])

    images[:, y1:y2, x1:x2, :] = shuffled_images[:, y1:y2, x1:x2, :]
    labels = lam * labels + (1 - lam) * shuffled_labels

    # return images, labels


    DIM = image.shape[0]
    CLASSES = label.shape[-1]
    
    # Get a random integer for the CutMix ratio
    mixup_ratio = tf.cast(tf.random.uniform([], 0, 1) < PROBABILITY, tf.int32)
    mixup_ratio = tf.cast(mixup_ratio, tf.float32)

    # Get random box
    cut_rat = tf.math.sqrt(1. - mixup_ratio)
    cut_w = tf.cast(DIM * cut_rat, tf.int32)
    cut_x = tf.random.uniform([], 0, DIM, dtype=tf.int32)
    cut_y = tf.random.uniform([], 0, DIM, dtype=tf.int32)
    
    bbx1 = tf.clip_by_value(cut_x - cut_w // 2, 0, DIM)
    bby1 = tf.clip_by_value(cut_y - cut_w // 2, 0, DIM)
    bbx2 = tf.clip_by_value(cut_x + cut_w // 2, 0, DIM)
    bby2 = tf.clip_by_value(cut_y + cut_w // 2, 0, DIM)

    # Mix the images
    image_a = image
    image_b = tf.reverse(image, axis=[0])
    
    new_image = image_a[:, bbx1:bbx2, bby1:bby2, :]
    new_image = tf.concat([new_image, image_b[:, :bbx1, :bby1, :]], axis=0)
    new_label = label * (1 - mixup_ratio) + label[::-1] * mixup_ratio

    return new_image, new_label
