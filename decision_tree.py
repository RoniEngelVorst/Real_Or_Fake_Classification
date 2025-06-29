import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


def run_decision_tree(X_train, y_train, X_test, y_test, use_pca=False):
    if use_pca:
        print("Applying PCA for Decision Tree...")
        pca = PCA(n_components=100)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- Decision Tree Report ({} PCA) ---".format("with" if use_pca else "without"))
    print("Best parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
