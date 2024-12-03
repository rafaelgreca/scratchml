from scratchml.utils import split_data_into_batches
from ..metrics import accuracy, recall, precision, f1_score, confusion_matrix
from scratchml.regularizations import l2
from scratchml.scalers import StandardScaler
import numpy as np
from typing import List, Union


class SVC:
    """
    Support Vector Classifier (SVC) with options for linear, polynomial, and RBF kernels.
    """

    def __init__(
        self,
        C: float = 0.35,
        alpha: float = 0.01,
        kernel: str = "linear",
        degree: int = 3,
        max_iter: int = 1500,
        tol: float = 1e-4,
        learning_rate: float = 1e-5,
        decay: float = 0.999,
        batch_size: int = 64,
        early_stopping: bool = True,
        adaptive_lr: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        Creates a Support Vector Classifier (SVC) instance.

        Args:
            C (float): Regularization parameter. Defaults to 0.35.
            alpha (float): Learning rate for gradient descent. Defaults to 0.01.
            kernel (str): Kernel type to be used in the algorithm. Valid values are "linear", "polynomial", and "rbf". Defaults to "linear".
            degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Defaults to 3.
            max_iter (int): Maximum number of iterations for training. Defaults to 1500.
            tol (float): Tolerance for stopping criteria. Defaults to 1e-4.
            learning_rate (float): Initial learning rate for gradient descent. Defaults to 1e-5.
            decay (float): Learning rate decay factor. Defaults to 0.999.
            batch_size (int): Size of mini-batches for stochastic gradient descent. Defaults to 64.
            early_stopping (bool): Whether to use early stopping to terminate training when validation score is not improving. Defaults to True.
            adaptive_lr (bool): Whether to use adaptive learning rate. Defaults to False.
            verbose (int): Level of verbosity. 0 means no information, 1 means convergence information, 2 means detailed information. Defaults to 0.
        """
        self.C = C
        self.alpha = alpha
        self.kernel = kernel
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.adaptive_lr = adaptive_lr
        self.verbose = verbose
        self.weights = None
        self.classes_ = None
        self.bias = 0
        self.scaler = StandardScaler()
        self._valid_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
        ]
        self._validate_parameters()

    def _apply_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Applies the selected kernel function.

        Args:
            X1 (np.ndarray): First input array.
            X2 (np.ndarray): Second input array.

        Returns:
            np.ndarray: Result of applying the kernel function.
        """
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "polynomial":
            return (1 + np.dot(X1, X2.T)) ** self.degree
        elif self.kernel == "rbf":
            gamma = 1 / X1.shape[1]
            return np.exp(
                -gamma * np.square(np.linalg.norm(X1[:, np.newaxis] - X2, axis=2))
            )

    def _validate_parameters(self) -> None:
        """
        Validates the parameters of the SVC instance.

        Raises:
            ValueError: If any parameter is invalid.
        """
        try:
            assert isinstance(self.C, float) and self.C > 0
        except AssertionError as error:
            raise ValueError(
                "Regularization parameter C must be a positive float."
            ) from error

        try:
            assert isinstance(self.alpha, float) and self.alpha > 0
        except AssertionError as error:
            raise ValueError("Learning rate alpha must be a positive float.") from error

        try:
            assert self.kernel in ["linear", "polynomial", "rbf"]
        except AssertionError as error:
            raise ValueError(
                "Kernel must be one of 'linear', 'polynomial', or 'rbf'."
            ) from error

        try:
            assert isinstance(self.degree, int) and self.degree > 0
        except AssertionError as error:
            raise ValueError("Degree must be a positive integer.") from error

        try:
            assert isinstance(self.max_iter, int) and self.max_iter > 0
        except AssertionError as error:
            raise ValueError(
                "Maximum number of iterations must be a positive integer."
            ) from error

        try:
            assert isinstance(self.tol, float) and self.tol > 0
        except AssertionError as error:
            raise ValueError("Tolerance must be a positive float.") from error

        try:
            assert isinstance(self.learning_rate, float) and self.learning_rate > 0
        except AssertionError as error:
            raise ValueError("Learning rate must be a positive float.") from error

        try:
            assert isinstance(self.decay, float) and 0 < self.decay <= 1
        except AssertionError as error:
            raise ValueError("Decay must be a float between 0 and 1.") from error

        try:
            assert isinstance(self.batch_size, int) and self.batch_size > 0
        except AssertionError as error:
            raise ValueError("Batch size must be a positive integer.") from error

        try:
            assert isinstance(self.early_stopping, bool)
        except AssertionError as error:
            raise ValueError("Early stopping must be a boolean.") from error

        try:
            assert isinstance(self.adaptive_lr, bool)
        except AssertionError as error:
            raise ValueError("Adaptive learning rate must be a boolean.") from error

        try:
            assert isinstance(self.weights, (type(None), np.ndarray))
        except AssertionError as error:
            raise ValueError("Weights must be None or a numpy array.") from error

        try:
            assert isinstance(self.classes_, (type(None), np.ndarray))
        except AssertionError as error:
            raise ValueError("Classes must be None or a numpy array.") from error

        try:
            assert isinstance(self.bias, (int, float))
        except AssertionError as error:
            raise ValueError("Bias must be an integer or float.") from error

        try:
            assert isinstance(self.scaler, StandardScaler)
        except AssertionError as error:
            raise ValueError("Scaler must be an instance of StandardScaler.") from error

        try:
            assert isinstance(self._valid_metrics, list) and all(
                isinstance(metric, str) for metric in self._valid_metrics
            )
        except AssertionError as error:
            raise ValueError("Valid metrics must be a list of strings.") from error

    def _check_is_fitted(self) -> None:
        """
        Checks if the model is fitted.

        Raises:
            ValueError: If the model is not trained.
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the SVC model using mini-batch gradient descent.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """
        # Identify unique classes in the target labels
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            # Convert binary class labels to -1 and 1
            y_ = np.where(y == self.classes_[0], -1, 1)
        else:
            y_ = y.copy()

        # Standardize the features
        X = self.scaler.fit_transform(X)
        _, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)

        no_improvement_count = 0
        for iteration in range(self.max_iter):
            # Adjust learning rate based on decay
            lr = self.learning_rate * (self.decay**iteration)
            avg_update_norm = 0
            # Split data into mini-batches and shuffle
            for X_batch, y_batch in split_data_into_batches(
                X, y_, self.batch_size, shuffle=True
            ):
                weight_update = np.zeros_like(self.weights)
                bias_update = 0
                # Iterate over each instance in the mini-batch
                for idx in range(X_batch.shape[0]):
                    instance = X_batch[idx : idx + 1]
                    margin = y_batch[idx] * (
                        np.dot(instance.flatten(), self.weights) - self.bias
                    )

                    # Update weights and bias if the margin condition is not satisfied
                    if margin < 1:
                        weight_update += -self.C * y_batch[idx] * instance.flatten()
                        bias_update += -self.C * y_batch[idx]

                # Apply L2 regularization to the weight update
                weight_update += self.alpha * l2(self.weights)

                # Adjust learning rate if adaptive learning rate is enabled
                lr_adjusted = (
                    lr / (1 + np.linalg.norm(weight_update)) if self.adaptive_lr else lr
                )
                # Update weights and bias
                self.weights -= lr_adjusted * weight_update
                self.bias -= lr_adjusted * bias_update
                avg_update_norm += np.linalg.norm(weight_update)

            # Calculate average norm of weight updates
            avg_update_norm /= X.shape[0] / self.batch_size
            # Check for early stopping based on tolerance
            if self.early_stopping and avg_update_norm < self.tol:
                no_improvement_count += 1
                if (
                    no_improvement_count > 10
                ):  # Terminate if no improvement for 10 iterations
                    if self.verbose > 0:
                        print(f"Converged after {iteration} iterations.")
                    break
            else:
                no_improvement_count = 0

            # Print detailed information every 100 iterations if verbose level is 2
            if self.verbose == 2 and iteration % 100 == 0:
                print(
                    f"Iteration {iteration}: Avg norm of batch updates = {avg_update_norm}"
                )

        # Print final weight norm and bias if verbose level is greater than 0
        if self.verbose > 0:
            print(f"Final weight norm: {np.linalg.norm(self.weights)}")
            print(f"Final bias: {self.bias}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            ValueError: If the model is not trained.
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction.")
        X = self.scaler.transform(X)
        linear_output = np.dot(X, self.weights) - self.bias
        predictions = np.sign(linear_output)

        if len(self.classes_) == 2:
            return np.where(predictions == -1, self.classes_[0], self.classes_[1])
        else:
            return predictions

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
        labels_cm: List[int] = None,
        normalize_cm: bool = False,
    ) -> Union[np.float64, np.ndarray]:
        """
        Calculates the score of the model on a given dataset using the specified metric.

        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).
            metric (str): Metric to use for evaluation ("accuracy", "precision", "recall", "f1_score", "confusion_matrix"). Defaults to "accuracy".
            labels_cm (List[int], optional): Labels for confusion matrix computation, ignored for other metrics. Defaults to None.
            normalize_cm (bool, optional): Whether to normalize the confusion matrix. Defaults to False.

        Returns:
            Union[float, np.ndarray]: Computed score based on the specified metric.

        Raises:
            ValueError: If the specified metric is not valid.
        """
        self._check_is_fitted()
        if metric not in self._valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of {self._valid_metrics}.")

        predictions = self.predict(X)

        if metric == "accuracy":
            return accuracy(y, predictions)
        elif metric == "precision":
            return precision(y, predictions)
        elif metric == "recall":
            return recall(y, predictions)
        elif metric == "f1_score":
            return f1_score(y, predictions)
        elif metric == "confusion_matrix":
            return confusion_matrix(
                y, predictions, labels=labels_cm, normalize=normalize_cm
            )
