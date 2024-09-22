import numpy as np


def entropy(y: np.array) -> float:
    """
    Calculates the entropy classification criteria.

    Args:
        y (np.array): the target labels.

    Returns:
        float: the entropy value.
    """
    _, counts = np.unique(y, return_counts=True)
    norm_counts = counts / counts.sum()
    return (norm_counts * (1 - norm_counts)).sum()


def gini(y: np.array, epsilon: np.float32 = 1e-9) -> float:
    """
    Calculates the gini classification criteria.

    Args:
        y (np.array): the target labels.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        float: the entropy value.
    """
    _, counts = np.unique(y, return_counts=True)
    norm_counts = counts / counts.sum()
    norm_counts += epsilon
    return -(norm_counts * np.log(norm_counts)).sum()


def squared_error(y_mean: float, y: np.array) -> float:
    """
    Calculates the squared error (Mean Squared Error) regression criteria.

    Args:
        y_mean (float): the mean of the targets.
        y (np.array): the target labels.

    Returns:
        float: the squared error value.
    """
    return np.sum(np.power(y - y_mean, 2)) / len(y)
