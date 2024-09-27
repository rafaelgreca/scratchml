from ..utils import convert_array_numpy
import numpy as np


class PCA:
    """
    Creates a class for the Principal Component Analysis (PCA) dimension
    reduction method.
    """

    def __init__(self, n_components: int = None) -> None:
        """
        Creates a PCA instance.

        Args:
            n_components (int, optional): Number of components to keep. Defaults to None.
        """
        self.n_components = n_components
        self.components_ = None
        self.n_components_ = None
        self.n_samples_ = None
        self.n_features_in_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        self.covariance_matrix_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the PCA model.

        Args:
            X (np.ndarray): the features array.
        """
        X = convert_array_numpy(X)

        self.n_samples_, self.n_features_in_ = X.shape

        self._validate_parameters()

        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_in_) - 1
        else:
            self.n_components_ = self.n_components

        # getting the mean of the data samples
        self.mean_ = X.mean(axis=0)

        # normalizing/centering the data samples
        X -= self.mean_

        # getting the covariance matrix
        self.covariance_matrix_ = np.cov(X.T)

        # computing the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix_)

        # sorting eigenvectors from largest to lowest
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # getting the n largests components of the eigenvectors
        self.components_ = eigenvectors[: self.n_components_]
        self.explained_variance_ratio_ = (eigenvalues / sum(eigenvalues))[
            : self.n_components_
        ]
        self.explained_variance_ = eigenvalues[: self.n_components_]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Using the fitted PCA to transform a given set of features.

        Args:
            X (np.ndarray): the features array.

        Returns:
            np.ndarray: the new transformed features.
        """
        X -= self.mean_
        return X.dot(self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the PCA and then transforms the given set of features in sequence.

        Args:
            X (np.ndarray): the features array.

        Returns:
            np.ndarray: the new transformed features.
        """
        X = convert_array_numpy(X)

        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation (converts a transformed
        set of features to its original values).

        Args:
            X_transformed (np.ndarray): the transformed features array.

        Returns:
            np.ndarray: the original features array.
        """
        return X_transformed.dot(self.components_) + self.mean_

    def get_precision(self) -> np.ndarray:
        """
        Compute data precision matrix with the generative model.

        Returns:
            np.ndarray: the estimated precision of the data.
        """
        return np.linalg.inv(self.covariance_matrix_)

    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model.

        Returns:
            np.ndarray: the estimated covariance of the data.
        """
        return (self.components_.T * self.explained_variance_).dot(self.components_)

    def _validate_parameters(self) -> None:
        """
        Auxiliary function used to validate the values of the parameters
        passed during the initialization.
        """
        # validating the n_components value
        if self.n_components is not None:
            try:
                assert (isinstance(self.n_components, int)) and (
                    0 < self.n_components < self.n_features_in_
                )
            except AssertionError as error:
                raise ValueError(
                    "The 'n_components' value should be an integer, positive number "
                    + f"bigger than zero and smaller than {self.n_features_in_}.\n"
                ) from error
