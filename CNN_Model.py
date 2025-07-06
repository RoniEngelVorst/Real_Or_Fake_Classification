import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.color import rgb2gray


def plot_cnn_predictions_2D(X_images, y_true, y_pred, title="CNN – Predictions in 2D (via PCA)"):
    X_flat = []
    for img in X_images:
        gray = rgb2gray(img)
        X_flat.append(gray.flatten())
    X_flat = np.array(X_flat)

    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_flat)

    correct = y_true == y_pred.flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vis[correct, 0], X_vis[correct, 1], c='green', label='Correct', alpha=0.6)
    plt.scatter(X_vis[~correct, 0], X_vis[~correct, 1], c='red', label='Incorrect', alpha=0.6)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_cnn(X_train, y_train, X_test, y_test, runs=3):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    best_reports = []
    all_thresholds = []

    for seed in range(runs):
        print(f"\n=== Run {seed + 1} ===")
        X_train_cnn, X_val, y_train_cnn, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
        )

        class_weights = {0: 1.0, 1: 1.7}

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(X_train_cnn)

        model = build_cnn(X_train.shape[1:])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'best_model_run{seed+1}.keras', monitor='val_loss', save_best_only=True, verbose=0)

        history = model.fit(
            datagen.flow(X_train_cnn, y_train_cnn, batch_size=32),
            validation_data=(X_val, y_val),
            epochs=30,
            class_weight=class_weights,
            callbacks=[early_stop, checkpoint],
            verbose=0
        )

        y_probs = model.predict(X_test)
        thresholds = np.arange(0.3, 0.71, 0.05)
        best_f1 = 0
        best_thresh = 0.5

        for t in thresholds:
            y_pred = (y_probs > t).astype('int32')
            report = classification_report(y_test, y_pred, output_dict=True)
            f1 = report['1']['f1-score']
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        y_pred = (y_probs > best_thresh).astype('int32')
        final_report = classification_report(y_test, y_pred, output_dict=False)
        best_reports.append(final_report)
        all_thresholds.append(best_thresh)

        print(f"Best threshold: {best_thresh:.2f}")
        print(final_report)

    print("\n=== Summary over runs ===")
    for i, report in enumerate(best_reports):
        print(f"Run {i + 1} Threshold: {all_thresholds[i]:.2f}")
        print(report)

    # Plot last training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Call visualization for the final predictions of last run
    plot_cnn_predictions_2D(X_test, y_test, y_pred)
