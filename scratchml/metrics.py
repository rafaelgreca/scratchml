import numpy as np
from typing import Union

def mean_squared_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MSE, respectively.
    """
    if derivative:
        return (y_hat - y)
    else:
        return np.sum((y_hat - y) ** 2) / (2 * y.shape[0])

def root_mean_squared_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Root Mean Squared Error (RMSE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the RMSE, respectively.
    """
    if derivative:
        raise NotImplementedError
    else:
        return np.sqrt(np.sum((y_hat - y) ** 2) / (2 * y.shape[0]))
    
def mean_absolute_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Absolute Error (MAE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MAE, respectively.
    """
    if derivative:
        return (np.where(y_hat > y, 1, -1) / y.shape[0])
    else:
        return np.sum(np.abs(y - y_hat)) / (y.shape[0])
    
def median_absolute_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Median Absolute Error (MedAE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MedAE, respectively.
    """
    if derivative:
        return NotImplementedError
    else:
        return np.median(np.abs(y - y_hat)) / (y.shape[0])

def mean_absolute_percentage_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MAPE, respectively.
    """
    if derivative:
        raise NotImplementedError
    else:
        score = np.sum(np.abs((y - y_hat)/ np.max(epsilon, np.abs(y))))
        return  score / (y.shape[0])

def mean_squared_logarithmic_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
    epsilon: np.float32 = 1e-9
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Mean Squared Logarithmic Error (MSLE).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.
        epsilon (np.float32): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defaults to 1e-9.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the MSLE, respectively.
    """
    if derivative:
        raise NotImplementedError
    else:
        score = np.sum((np.log(1 + y + epsilon) - np.log(1 + y_hat + epsilon)) ** 2)
        return score / y.shape[0]

def max_error(
    y: np.ndarray,
    y_hat: np.ndarray,
    derivative: bool = False,
) -> Union[np.ndarray, np.float32]:
    """
    Calculates the Max Error (ME).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.
        derivative (bool, optional): whether to use the
            derivative function or not. Defaults to False.

    Returns:
        Union[np.ndarray, np.float32]: the derivative function
            or the value of the ME, respectively.
    """
    if derivative:
        raise NotImplementedError
    else:
        return np.abs(y - y_hat)
    
def r_squared(
    y: np.ndarray,
    y_hat: np.ndarray
) -> np.float32:
    """
    Calculates the R Squared (R2).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the R2 error.
    """
    # sum of the squared residuals
    u = ((y - y_hat)** 2).sum()

    # total sum of squares
    v = ((y - y_hat.mean()) ** 2).sum()

    return (1 - (u/v))

def accuracy(
    y: np.ndarray,
    y_hat: np.ndarray
) -> np.float32:
    """
    Calculates the Accuracy score.

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the Accuracy score.
    """
    score = (y == np.squeeze(y_hat)).astype(int)
    return np.sum(score) / y.shape[0]

def precision(
    y: np.ndarray,
    y_hat: np.ndarray
) -> np.float32:
    """
    Calculates the Precision score.

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the Precision score.
    """
    tp = y_hat[(y == 1) == 1].sum()
    fp = y[(y_hat == 1) == 0].sum()
    return tp / (tp + fp)

def recall(
    y: np.ndarray,
    y_hat: np.ndarray
) -> np.float32:
    """
    Calculates the Recall score.

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the Recall score.
    """
    tp = y_hat[(y == 1) == 1].sum()
    fn = y[(y_hat == 0) == 1].sum()
    return tp / (tp + fn)

def f1_score(
    y: np.ndarray,
    y_hat: np.ndarray
) -> np.float32:
    """
    Calculates the F1-Score (F1).

    Args:
        y (np.ndarray): the true value for y.
        y_hat (np.ndarray): the predicted value for y.

    Returns:
        np.float32: the value of the F1 Score.
    """
    op = (precision(y, y_hat) * recall(y, y_hat))
    div = (precision(y, y_hat) + recall(y, y_hat))
    return 2 * (op / div)