from ..scratchml.models.knn import KNNClassifier, KNNRegressor
from ..scratchml.utils import KFold
from sklearn.datasets import make_blobs, make_regression
from typing import Union
import numpy as np


def train_model(
    model: Union[KNNRegressor, KNNClassifier],
    X: np.ndarray,
    y: np.ndarray,
    metric: "str",
) -> None:
    """
    Auxiliary function to train the KNN models.

    Args:
        model (Union[KNNRegressor, KNNClassifier]): the model instance which will
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

        # assessing the model's performance
        score = model.score(X=X_test, y=y_test, metric=metric)

        print(score)


def example_knn() -> None:
    """
    Practical example of how to use the KNNClassifier and KNNRegressor models.
    """
    # generating a dataset for the classfication task
    X, y = make_blobs(n_samples=2000, n_features=4, centers=3, shuffle=True)

    # creating a KNN model for classification
    knn = KNNClassifier(
        n_neighbors=5, weights="uniform", p=2, metric="minkowski", n_jobs=None
    )

    train_model(model=knn, X=X, y=y, metric="accuracy")

    # generating a dataset for the regression task
    X, y = make_regression(
        n_samples=10000, n_features=5, n_targets=1, shuffle=True, noise=30
    )

    # creating a KNN model for regression
    knn_regression = KNNRegressor(
        n_neighbors=5, weights="uniform", p=2, metric="minkowski", n_jobs=None
    )

    train_model(model=knn_regression, X=X, y=y, metric="r_squared")


if __name__ == "__main__":
    example_knn()
