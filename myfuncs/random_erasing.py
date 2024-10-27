# random_erasing.py
# Random Erasing

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd

import tensorflow as tf

def random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    '''
    Randomly remove parts of the image to simulate occlusion.
    '''

    # Apply Random Erasing to the batch.
    for i in range(images.shape[0]):
        if np.random.uniform(0, 1) > probability:
            continue
        area = images.shape[1] * images.shape[2]
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1 / r1)

        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))

        if w < images.shape[2] and h < images.shape[1]:
            x1 = np.random.randint(0, images.shape[1] - h)
            y1 = np.random.randint(0, images.shape[2] - w)
            images[i, x1:x1 + h, y1:y1 + w, :] = np.random.uniform(0, 1, (h, w, images.shape[3]))

    # return images

    if np.random.rand() > probability:
        return image

    area = image.shape[0] * image.shape[1]
    target_area = np.random.uniform(sl, sh) * area
    aspect_ratio = np.random.uniform(r1, 1/r1)

    h = int(np.round(np.sqrt(target_area * aspect_ratio)))
    w = int(np.round(np.sqrt(target_area / aspect_ratio)))

    if w < image.shape[1] and h < image.shape[0]:
        x1 = np.random.randint(0, image.shape[0] - h)
        y1 = np.random.randint(0, image.shape[1] - w)
        image[x1:x1+h, y1:y1+w, :] = np.random.uniform(0, 1, size=(h, w, image.shape[2]))
    
    return image