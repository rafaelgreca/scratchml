import numpy as np

def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False
) -> np.float32:
    if derivative:
        return (y_hat - y)
    else:
        return np.sum((y_hat - y) ** 2) / (2 * y.shape[0])