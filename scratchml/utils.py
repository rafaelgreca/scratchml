import numpy as np
from typing import Any

def convert_array_numpy(
    array: Any
) -> np.ndarray:
    """
    Auxiliary function that converts an array to numpy array.

    Args:
        array (Any): the array that will be converted.

    Returns:
        array (np.ndarray): the converted numpy array.
    """
    if isinstance(array, list):
        array = np.asarray(array, dtype="O")
        return array
    if isinstance(array, np.ndarray):
        return array
    else:
        raise TypeError("Invalid type. Should be np.ndarray or list.\n")