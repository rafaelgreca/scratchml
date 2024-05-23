import numpy as np

def binary_cross_entropy(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9
) -> np.ndarray:
    if derivative:
        return (y_hat - y)
    else:
        y1 = (y * np.log(y_hat) + epsilon)
        y2 = ((1 - y) * np.log(1 - y_hat + epsilon))
        return (-1 * (1/y.shape[0])) * np.sum(y1, y2)