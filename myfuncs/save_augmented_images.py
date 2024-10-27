# save_augmented_images.py
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import os
import cv2
import glob
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
# from myfuncs.AlbumentationsImageDataGenerator import AlbumentationsImageDataGenerator
from .AlbumentationsImageDataGenerator import AlbumentationsImageDataGenerator

def save_augmented_images(file_paths, augmentations, image_size=(224, 224), num_augmentations=3, augmented_dir=None):
    for file_path in tqdm(file_paths):
        # Extract the car model name from the file path
        car_model_name = os.path.basename(os.path.dirname(file_path))

        # Special case handling to prevent creation of incorrect folders
        if "Ram C" in car_model_name and "V Cargo Van Minivan 2012" not in car_model_name:
            car_model_name = "Ram C-V Cargo Van Minivan 2012"

        car_model_augmented_path = os.path.join(augmented_dir, car_model_name)

        # Create the augmented directory if it doesn't exist
        if not os.path.exists(car_model_augmented_path):
            os.makedirs(car_model_augmented_path)

        # Initialize the data generator for the current car model
        data_generator = AlbumentationsImageDataGenerator([file_path], augmentations, image_size)

        for i in range(num_augmentations):
            augmented_image = data_generator[0]  # Only one image per file_path in this context
            if augmented_image is not None:
                # Save the augmented image
                original_filename = os.path.basename(file_path)
                augmented_filename = f"aug_{i}_{original_filename}"
                augmented_image_path = os.path.join(car_model_augmented_path, augmented_filename)
                image_to_save = augmented_image  # Already in uint8 format
                cv2.imwrite(augmented_image_path, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))

    print("Augmented images saved successfully.")

    # Check and delete the incorrect "Ram C" folder if it exists
    incorrect_folder = os.path.join(augmented_dir, "Ram C")
    if os.path.exists(incorrect_folder):
        shutil.rmtree(incorrect_folder)
        print(f"Incorrect folder '{incorrect_folder}' has been deleted.")

