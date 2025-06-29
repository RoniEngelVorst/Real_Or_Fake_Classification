import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split


def load_logo_dataset(base_dir, csv_file="file_mapping.csv", image_size=(128, 128), test_size=0.2, random_state=42):
    """
    Loads logo images and labels from a structured dataset based on a CSV mapping.

    Args:
        base_dir (str): Root directory where images and CSV file are located.
        csv_file (str): Name of the CSV file with Filename and Label columns.
        image_size (tuple): Resize dimensions for each image.
        test_size (float): Proportion of test data in the split.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets with images and labels.
    """
    csv_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in df.iterrows():
        image_path = os.path.join(base_dir, row["Filename"])
        try:
            img = Image.open(image_path).convert("RGB").resize(image_size)
            images.append(np.array(img))
            labels.append(1 if row["Label"].strip().lower() == "genuine" else 0)
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")

    X = np.array(images)
    y = np.array(labels)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
