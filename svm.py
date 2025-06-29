import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


def run_svm(X_train, y_train, X_test, y_test, use_pca=False):
    if use_pca:
        print("Applying PCA for SVM...")
        pca = PCA(n_components=100)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- SVM Report ({} PCA) ---".format("with" if use_pca else "without"))
    print("Best parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))