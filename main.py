import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt


def plot_predictions_2D(X_high_dim, y_true, y_preds_dict):
    """ Projects high-dimensional data to 2D using PCA, and plots predictions for multiple models. """
    # Project data to 2D using PCA
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_high_dim)

    for model_name, y_pred in y_preds_dict.items():
        correct = y_true == y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(X_vis[correct, 0], X_vis[correct, 1], c='green', label='Correct', alpha=0.6)
        plt.scatter(X_vis[~correct, 0], X_vis[~correct, 1], c='red', label='Incorrect', alpha=0.6)
        plt.title(f"{model_name} – Predictions in 2D (via PCA)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



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
    X_train_raw, X_test_raw, y_train, y_test, test_file_names = load_logo_dataset(base_dir="data", return_file_names=True)

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
    y_pred_perceptron = run_perceptron(X_train_scaled, y_train, X_test_scaled, y_test)
    y_pred_svm = run_svm(X_train_scaled, y_train, X_test_scaled, y_test)
    y_pred_tree = run_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test)

    # === Apply PCA ===
    print("\n--- With PCA (100 components) ---")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # === Run again using PCA ===
    run_perceptron(X_train_pca, y_train, X_test_pca, y_test)
    run_svm(X_train_pca, y_train, X_test_pca, y_test)
    run_decision_tree(X_train_pca, y_train, X_test_pca, y_test)

    # === Run CNN ===
    print("\n--- CNN ---")
    run_cnn(X_train_raw, y_train, X_test_raw, y_test)

    # === Load file mapping and analyze accuracy by brand ===
    print("\n--- Brand-level analysis ---")
    df = pd.read_csv("data/file_mapping.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    results = pd.DataFrame({
        "file_name": test_file_names,
        "true_label": y_test,
        "perceptron": y_pred_perceptron,
        "svm": y_pred_svm,
        "tree": y_pred_tree
    })
    merged = results.merge(df, left_on="file_name", right_on="filename")

    for model in ["perceptron", "svm", "tree"]:
        print(f"\n--- {model.capitalize()} accuracy by brand ---")
        acc = (merged["true_label"] == merged[model]).groupby(merged["brand_name"]).mean()
        print("Best:")
        print(acc.sort_values(ascending=False).head(5))
        print("Worst:")
        print(acc.sort_values().head(5))

    # === Visualization: Show prediction correctness in 2D for all models ===
    plot_predictions_2D(
        X_test_scaled,
        y_test,
        {
            "Perceptron": y_pred_perceptron,
            "SVM": y_pred_svm,
            "Decision Tree": y_pred_tree
        }
    )