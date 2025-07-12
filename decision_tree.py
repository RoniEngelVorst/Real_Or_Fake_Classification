from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def run_decision_tree(X_train, y_train, X_test, y_test):

    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- Decision Tree Report ---")
    print("Best parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    return y_pred

