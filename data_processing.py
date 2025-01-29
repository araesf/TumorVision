import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class TumorDataProcessor:
    def __init__(self, root_dir, labels, img_size=200):
        self.root_dir = root_dir
        self.labels = labels
        self.img_size = img_size
        self.training_data = []
        self.validation_data = []

    def create_training_data(self):
        training_dir = os.path.join(self.root_dir, "Training")
        for label in self.labels:
            class_dir = os.path.join(training_dir, label)
            if os.path.isdir(class_dir):
                for brain_scan in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, brain_scan)
                    cv_image = cv2.imread(img_path)
                    if cv_image is not None:
                        modified_cv_image = cv2.resize(cv_image, (self.img_size, self.img_size))
                        self.training_data.append([modified_cv_image, label])
                        print(f'Added scan to training list: {brain_scan}')
                    else:
                        print(f'Failed to load image: {brain_scan}')

    def create_validation_data(self):
        testing_dir = os.path.join(self.root_dir, "Testing")
        for label in self.labels:
            class_dir = os.path.join(testing_dir, label)
            if os.path.isdir(class_dir):
                for brain_scan in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, brain_scan)
                    cv_image = cv2.imread(img_path)
                    if cv_image is not None:
                        modified_cv_image = cv2.resize(cv_image, (self.img_size, self.img_size))
                        self.validation_data.append([modified_cv_image, label])
                        print(f'Added scan to validation list: {brain_scan}')
                    else:
                        print(f'Failed to load image: {brain_scan}')
    
    def get_training_data(self):
        return self.training_data

    def get_validation_data(self):
        return self.validation_data


def convert_to_tensor(data):
    X = []
    y = []
    for img, label in data:
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        X.append(img)

        if label in ["glioma", "meningioma", "pituitary"]:
            y.append(1)  # Tumor
        else:
            y.append(0)  # No tumor

    
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)

    X_tensor = torch.tensor(X_array)
    y_tensor = torch.tensor(y_array)

    return TensorDataset(X_tensor, y_tensor)


def load_data(root_dir, labels):
    # Initialize the data processor
    processor = TumorDataProcessor(root_dir=root_dir, labels=labels, img_size=200)
    
    # Create the training and validation data separately
    processor.create_training_data()
    processor.create_validation_data()

    # Get the processed training and validation data
    training_data = processor.get_training_data()
    validation_data = processor.get_validation_data()

    # Convert the datasets into tensor datasets
    train_dataset = convert_to_tensor(training_data)
    val_dataset = convert_to_tensor(validation_data)

    return train_dataset, val_dataset
