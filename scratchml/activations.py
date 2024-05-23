import numpy as np

def sigmoid(
    x: np.ndarray,
    epsilon: np.float32 = 1e-9
) -> np.ndarray:
    """
    Applies the Sigmoid activation function.

    Args:
        x (np.ndarray): the features array.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        np.ndarray: the output of the sigmoid function
            for the given numpy array.
    """
    return 1.0 / (1.0 + np.exp(-1 * x + epsilon))