from scratchml.models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from scratchml.utils import KFold
from sklearn.datasets import make_regression, make_classification
from typing import Union
import numpy as np


def train_model(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    X: np.ndarray,
    y: np.ndarray,
    metric: "str",
) -> None:
    """
    Auxiliary function to train the Decision Tree models.

    Args:
        model (Union[DecisionTreeClassifier, DecisionTreeRegressor]): the model instance which will
            be trained.
        X (np.ndarray): the feature array.
        y (np.ndarray): the target/classes array.
        metric (str): which metric will be used to assess the model's performance.
    """
    # splitting the data into training and testing using KFold
    folds = KFold(X, y, stratify=True, shuffle=True, n_splits=5)

    for train_indexes, test_indexes in folds:
        # getting the training and test sets
        X_train = X[train_indexes]
        y_train = y[train_indexes]

        X_test = X[test_indexes]
        y_test = y[test_indexes]

        # fitting the model
        model.fit(X=X_train, y=y_train)

        # model.print_tree(model.tree_)

        # assessing the model's performance
        score = model.score(X=X_test, y=y_test, metric=metric)

        print(score)


def example_decision_tree() -> None:
    """
    Practical example of how to use the DecisionTreeClassifier and
    DecisionTreeRegressor models.
    """
    # generating a dataset for the classfication task
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=2, shuffle=True)

    # creating a KNN model for classification
    dt = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=3)

    train_model(model=dt, X=X, y=y, metric="accuracy")

    # generating a dataset for the regression task
    X, y = make_regression(
        n_samples=2000, n_features=4, n_targets=1, shuffle=True, noise=0, coef=False
    )

    # creating a KNN model for regression
    dt_regression = DecisionTreeRegressor(
        criterion="squared_error", max_depth=100, min_samples_split=3
    )

    train_model(model=dt_regression, X=X, y=y, metric="r_squared")


if __name__ == "__main__":
    example_decision_tree()
