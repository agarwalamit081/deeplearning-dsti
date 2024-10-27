# AlbumentationsImageDataGenerator.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import os
import albumentations as A
import tensorflow as tf
import cv2
from tqdm import tqdm


class AlbumentationsImageDataGenerator:
    def __init__(self, file_paths, labels, bboxes, augmentations=None, image_size=(224, 224), mixup=False, cutmix=False, random_erasing=False, apply_clahe=False, apply_hist_eq=False):
        self.file_paths = file_paths
        self.labels = labels
        self.bboxes = bboxes
        self.augmentations = augmentations
        self.image_size = image_size
        self.mixup = mixup
        self.cutmix = cutmix
        self.random_erasing = random_erasing
        self.apply_clahe = apply_clahe
        self.apply_hist_eq = apply_hist_eq

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = self.__load_image(file_path)
        if image is None:
            return None

        if self.augmentations:
            image = self.augmentations(image=image)['image']

        return image

    def __load_image(self, file_path):
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None

        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to read image: {file_path}")
            return None

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
        except Exception as e:
            print(f"Error processing image {file_path}: {str(e)}")
            return None

        # Return as uint8 to ensure compatibility with Albumentations
        return image
        # return image / 255.0
