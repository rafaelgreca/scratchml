import numpy as np

def sigmoid(
    x: np.ndarray,
    epsilon: np.float32 = 1e-9
) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-1 * x + epsilon))