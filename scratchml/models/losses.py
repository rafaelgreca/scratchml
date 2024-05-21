import numpy as np

def r_squared(
    y: np.ndarray,
    y_hat: np.ndarray
) -> np.float32:
    # sum of the squared residuals
    u = ((y - y_hat)** 2).sum()

    # total sum of squares
    v = ((y - y_hat.mean()) ** 2).sum()

    return (1 - (u/v))

def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False
) -> np.float32:
    if derivative:
        return (y_hat - y)
    else:
        return np.sum((y_hat - y) ** 2) / (2 * y.shape[0])

def binary_cross_entropy(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: float = 1e-9
) -> np.ndarray:
    if derivative:
        return (y_hat - y)
    else:
        y1 = (y * np.log(y_hat) + epsilon)
        y2 = ((1 - y) * np.log(1 - y_hat + epsilon))
        return (-1 * (1/y.shape[0])) * np.sum(y1, y2)