import numpy as np

def linear_kernel(X1, X2):
    """
    Computes the linear kernel between two sets of data points.

    Args:
        X1 (np.ndarray): First set of data points.
        X2 (np.ndarray): Second set of data points.

    Returns:
        np.ndarray: Linear kernel matrix.
    """
    return np.dot(X1, X2.T)

def polynomial_kernel(X1, X2, degree=3, coef0=1):
    """
    Computes the polynomial kernel between two sets of data points.

    Args:
        X1 (np.ndarray): First set of data points.
        X2 (np.ndarray): Second set of data points.
        degree (int, optional): Degree of the polynomial. Defaults to 3.
        coef0 (float, optional): Independent term in polynomial kernel. Defaults to 1.

    Returns:
        np.ndarray: Polynomial kernel matrix.
    """
    return (np.dot(X1, X2.T) + coef0) ** degree

def rbf_kernel(X1, X2, gamma="scale"):
    """
    Computes the Radial Basis Function (RBF) kernel between two sets of data points.

    Args:
        X1 (np.ndarray): First set of data points.
        X2 (np.ndarray): Second set of data points.
        gamma (str or float, optional): Kernel coefficient. Defaults to "scale".

    Returns:
        np.ndarray: RBF kernel matrix.
    """
    if gamma == "scale":
        gamma = 1.0 / X1.shape[1]
    elif gamma == "auto":
        gamma = 1.0

    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    return np.exp(-gamma * (X1_norm + X2_norm - 2 * np.dot(X1, X2.T)))
