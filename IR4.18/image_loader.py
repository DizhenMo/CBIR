import os
import numpy as np
from PIL import Image


def load_images(folder_path):
    images = []
    labels = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(class_folder)
    return images, labels