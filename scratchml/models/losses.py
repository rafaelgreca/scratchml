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