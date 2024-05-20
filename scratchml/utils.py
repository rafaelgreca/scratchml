import numpy as np

def convert_array_numpy(array):
    if isinstance(array, list):
        array = np.asarray(array)
        return array
    if isinstance(array, np.ndarray):
        return array
    else:
        return TypeError