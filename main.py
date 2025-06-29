import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from perceptron import run_perceptron
from svm import run_svm
from decision_tree import run_decision_tree
from CNN_Model import run_cnn
from skimage.feature import hog
from skimage.color import rgb2gray
from data_loader import load_logo_dataset


def extract_hog_features(images):
    """Extracts HOG features from a list of images."""
    features = []
    for img in images:
        gray = rgb2gray(img)
        hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)


if __name__ == "__main__":
    # === Load dataset from CSV and image folders ===
    print("Loading dataset from file_mapping.csv...")
    X_train_raw, X_test_raw, y_train, y_test = load_logo_dataset(base_dir="data")

    # === Extract HOG features ===
    print("Extracting HOG features...")
    X_train_hog = extract_hog_features(X_train_raw)
    X_test_hog = extract_hog_features(X_test_raw)

    # === Normalize features ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hog)
    X_test_scaled = scaler.transform(X_test_hog)

    # === Run classical models without PCA ===
    print("\n--- Without PCA ---")
    run_perceptron(X_train_scaled, y_train, X_test_scaled, y_test)
    run_svm(X_train_scaled, y_train, X_test_scaled, y_test)
    run_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test)

    # === Apply PCA ===
    print("\n--- With PCA (100 components) ---")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    run_perceptron(X_train_pca, y_train, X_test_pca, y_test)
    run_svm(X_train_pca, y_train, X_test_pca, y_test)
    run_decision_tree(X_train_pca, y_train, X_test_pca, y_test)

    # === Run CNN if needed ===
    # print("\n--- CNN ---")
    # run_cnn(X_train_raw, y_train, X_test_raw, y_test)
