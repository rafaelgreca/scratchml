import numpy as np

def accuracy(
    y: np.ndarray,
    y_hat: np.ndarray
) -> np.float32:
    score = (y == np.squeeze(y_hat)).astype(int)
    return np.sum(score) / y.shape[0]