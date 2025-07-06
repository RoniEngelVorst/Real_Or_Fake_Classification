import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


def run_perceptron(X_train, y_train, X_test, y_test):

    param_grid = {
        'penalty': [None, 'l2', 'l1'],
        'alpha': [0.0001, 0.001, 0.01]
    }

    grid = GridSearchCV(Perceptron(), param_grid, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- Perceptron Report ---")
    print("Best parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    return y_pred
