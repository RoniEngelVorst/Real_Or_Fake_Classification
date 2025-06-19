import os
from Pillow import Image
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.color import rgb2gray


def load_images_from_folder(folder_path, label, image_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(image_size)
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    return images, labels


def load_dataset(base_path="data"):
    train_fake, y_fake = load_images_from_folder(os.path.join(base_path, "train/Fake"), 0)
    train_real, y_real = load_images_from_folder(os.path.join(base_path, "train/Genuine"), 1)
    test_fake, y_test_fake = load_images_from_folder(os.path.join(base_path, "test/Fake"), 0)
    test_real, y_test_real = load_images_from_folder(os.path.join(base_path, "test/Genuine"), 1)

    X_train = np.array(train_fake + train_real)
    y_train = np.array(y_fake + y_real)
    X_test = np.array(test_fake + test_real)
    y_test = np.array(y_test_fake + y_test_real)

    return X_train, y_train, X_test, y_test


def extract_hog_features(images):
    features = []
    for img in images:
        gray = rgb2gray(img)
        hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()
    print(f"Loaded {len(X_train)} training images and {len(X_test)} testing images")

    print("Extracting HOG features...")
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)
    print("Feature extraction completed.")