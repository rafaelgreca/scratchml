import numpy as np
from typing import Any, Union, Tuple

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: Union[float, int] = None,
    train_size: Union[float, int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): the features array.
        y (np.ndarray): the labels array.
        test_size (Union[float, int], optional): the test set size
            (in total samples or the ratio of the entire dataset). Defaults to None.
        train_size (Union[float, int], optional): the train set size
            (in total samples or the ratio of the entire dataset). Defaults to None.
        shuffle (bool, optional): whether to shuffle the data or not. Defaults to True.
        stratify (bool, optional): whether to stratify the set based on
            the labels (in other words, split the data while mantaining the classes
            distribution )or not. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: the X train,
            X test, y train, and y test sets, respectively.
    """
    X = convert_array_numpy(X)
    y = convert_array_numpy(y)
    
    # validating the train_size and test_size parameters
    # as just one of them should be used
    try:
        assert not ((train_size == None) and (test_size == None)) or\
                ((train_size != None) and (test_size != None))
    except AssertionError:
        raise RuntimeError(
            f"You should pass train_size or test_size, not both or neither.\n"
        )

    # validating the test size parameter
    if test_size != None:
        if isinstance(test_size, float):
            try:
                assert 0 < test_size < 1
                test_split_ratio = test_size
                
            except AssertionError:
                raise ValueError("Test size value should be between 0 and 1.\n")
        elif isinstance(test_size, int):
            try:
                assert 0 < test_size < X.shape[0]
                test_split_ratio = test_size / X.shape[0]
            except AssertionError:
                raise ValueError(
                    f"Test size value should be between 0 and {X.shape[0]}.\n"
                )
    
    # validating the train size parameter
    if train_size != None:
        if isinstance(train_size, float):
            try:
                assert 0 < train_size < 1
                train_split_ratio = train_size
            except AssertionError:
                raise ValueError("Train size value should be between 0 and 1.\n")
        elif isinstance(train_size, int):
            try:
                assert 0 < train_size < X.shape[0]
                train_split_ratio = train_size / X.shape[0]
            except AssertionError:
                raise ValueError(
                    f"Train size value should be between 0 and {X.shape[0]}.\n"
                )
    
    # defining the split ratio of the train set
    if train_size == None:
        train_split_ratio = 1 - test_split_ratio

    # shuffling the arrays
    if shuffle:
        shuffled_indices = np.arange(X.shape[0])
        np.random.shuffle(shuffled_indices)

        X = X[shuffled_indices]
        y = y[shuffled_indices]
    
    if stratify:
        # analysing the classes distribution
        unique, counts = np.unique(y, return_counts=True)
        counts = np.asarray((unique, counts)).T
        classes_distribution = [
            (u, c)
            for u, c in counts
        ]

        train_indexes = []
        test_indexes = []

        # getting the indexes of each class considering how many
        # times each one occurred on the sample and splitting it
        # using the train test ratio
        # e.g.: [[0 100], [1 200]] => classes distribuition: 66%
        # class 1 and 33% class 0. If we want the train set to be
        # composed of 80% of the data, so we will have 158 (0.8 * 0.6 * 300)
        # samples for class 1 and 79 (0.8 * 0.33 * 300) for class 0
        for c, d in classes_distribution:
            _y = np.argwhere(y == c).reshape(-1)
            _size = int(d * train_split_ratio)

            train_indexes.extend(_y[:_size])
            test_indexes.extend(_y[_size:])

        X_train = X[train_indexes]
        X_test = X[test_indexes]

        y_train = y[train_indexes]
        y_test = y[test_indexes]

    else:
        # splitting the arrays sequentially
        train_indexes = int(train_split_ratio * X.shape[0])

        X_train = X[:train_indexes]
        X_test = X[train_indexes:]

        y_train = y[:train_indexes]
        y_test = y[train_indexes:]

    # converting the train and test sets to numpy arrays
    X_train = convert_array_numpy(X_train)
    X_test = convert_array_numpy(X_test)
    y_train = convert_array_numpy(y_train)
    y_test = convert_array_numpy(y_test)
    
    return X_train, X_test, y_train, y_test

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