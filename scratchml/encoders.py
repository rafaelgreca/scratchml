import numpy as np
from abc import ABC
from typing import List

class BaseEncoder(ABC):
    """
    Encoders base class.
    """
    def __init__(self) -> None:
        pass

    def fit(
        self,
        y: np.ndarray
    ) -> None:
        pass

    def transform(
        self,
        y: np.ndarray
    ) -> np.ndarray:
        pass

    def fit_transform(
        self,
        y: np.ndarray
    ) -> np.ndarray:
        pass

    def inverse_transform(
        self,
        y: np.ndarray
    ) -> np.ndarray:
        pass

class LabelEncoder(BaseEncoder):
    def __init__(self) -> None:
        """
        Creates a LabelEncoder's instance.
        """
        self.classes_ = None
        self.classes_map_ = None
    
    def fit(
        self,
        y: np.ndarray
    ) -> None:
        """
        Fits the LabelEncoder.

        Args:
            y (np.array): the classes array. Defaults to None.
        """
        if not (isinstance(y, np.ndarray) or isinstance(y, List)):
            return TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        self.classes_ = np.sort(np.unique(y))
        self.classes_map_ = {c: i for i, c in enumerate(self.classes_)}

    def transform(
        self,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Using the fitted LabelEncoder to encode the classes.

        Args:
            y (np.array): the classes array.

        Returns:
            y (np.ndarray): the encoded classes array.
        """
        if not (isinstance(y, np.ndarray) or isinstance(y, List)):
            return TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        return np.array([self.classes_map_[v] for v in y])

    def fit_transform(
        self,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Fits the LabelEncoder and then transforms the given set of classes in sequence.

        Args:
            y (np.array): the classes array.

        Returns:
            np.ndarray: the encoded classes array.
        """
        if not (isinstance(y, np.ndarray) or isinstance(y, List)):
            return TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        self.fit(y)
        return self.transform(y)

    def inverse_transform(
        self, y: np.ndarray
    ) -> np.ndarray:
        """
        Applies the inverse transformation (converts a encoded
        set of classes to its original values).

        Args:
            y (np.ndarray): the encoded classes array.

        Returns:
            np.ndarray: the original classes array.
        """
        if not (isinstance(y, np.ndarray) or isinstance(y, List)):
            return TypeError(f"Expected type np.ndarray or list, got {type(y)}.\n")

        inverse_classes_map = dict(map(reversed, self.classes_map_.items()))
        return np.array([inverse_classes_map[v] for v in y])