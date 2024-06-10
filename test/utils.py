import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
from typing import Tuple, Union, Callable


def repeat(times: int) -> Callable:
    """
    Auxiliary function used in unittest to repeat a specific test for n times.

    Args:
        times (int): the number of times the test should be repeated.
    """

    def repeatHelper(f):
        def callHelper(*args):
            for _ in range(0, times):
                f(*args)

        return callHelper

    return repeatHelper


def generate_blob_dataset(
    n_samples: int = 10000, n_features: int = 5, centers: int = 2, shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auxiliary function used to create a blob dataset.

    Args:
        n_samples (int, optional): the number of samples. Defaults to 10000.
        n_features (int, optional): the number of features. Defaults to 5.
        centers (int, optional): the number of centers. Defaults to 2.
        shuffle (bool, optional): whether to shuffle the data or not. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the features and classes arrays, respectively.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        shuffle=shuffle,
        return_centers=False,
    )

    return X, y


def generate_classification_dataset(
    n_samples: int = 10000,
    n_features: int = 5,
    n_classes: int = 2,
    n_clusters_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auxiliary function used to create a classification dataset.

    Args:
        n_samples (int, optional): the number of samples. Defaults to 10000.
        n_features (int, optional): the number of features. Defaults to 5.
        n_classes (int, optional): the number of classes. Defaults to 2.
        n_clusters_per_class (int, optional): the number of clusters per class. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the features and classes arrays, respectively.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
    )
    return X, y


def generate_regression_dataset(
    n_samples: int = 10000,
    n_features: int = 1,
    n_targets: int = 1,
    shuffle: bool = True,
    noise: int = 20,
    coef: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Auxiliary function used to create a regression dataset.

    Args:
        n_samples (int, optional): the number of samples. Defaults to 10000.
        n_features (int, optional): the number of features. Defaults to 1.
        n_targets (int, optional): the number of targets. Defaults to 1.
        shuffle (bool, optional): whether to shuffle the data or not. Defaults to True.
        noise (int, optional): how much noise should be added to the data. Defaults to 20.
        coef (bool, optional): whether to return the coefficient or not. Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            the features, targets, and coefficient (when needed) arrays, respectively.
    """
    if coef:
        X, y, coef = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_targets,
            shuffle=shuffle,
            noise=noise,
            coef=coef,
        )
        return X, y, coef

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        shuffle=shuffle,
        noise=noise,
        coef=coef,
    )
    return X, y
