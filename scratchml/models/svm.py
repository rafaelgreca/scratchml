from abc import ABC
from cvxopt import matrix, solvers
from scratchml.utils import split_data_into_batches
from scratchml.regularizations import l2
from scratchml.scalers import StandardScaler
from typing import List, Union
from ..metrics import (
    accuracy,
    recall,
    precision,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    root_mean_squared_error,
    r_squared,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    max_error,
)
from ..kernels import rbf_kernel, linear_kernel, polynomial_kernel
from ..utils import convert_array_numpy
import numpy as np


class SVMBase(ABC):
    """
    Creates a base class for the Support Vector Machine (SVM) model.
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
        gamma: str = "scale",
    ) -> None:
        """
        Creates a Support Vector Classifier (SVC) instance.

        Args:
            C (float): Regularization parameter. Defaults to 0.35.
            alpha (float): Learning rate for gradient descent. Defaults to 0.01.
            kernel (str): Kernel type to be used in the algorithm. Valid values are "linear",
                "polynomial", and "rbf". Defaults to "linear".
            degree (int): Degree of the polynomial kernel function ('poly').
                Ignored by other kernels. Defaults to 3.
            max_iter (int): Maximum number of iterations for training. Defaults to 1500.
            tol (float): Tolerance for stopping criteria. Defaults to 1e-4.
            learning_rate (float): Initial learning rate for gradient descent. Defaults to 1e-5.
            decay (float): Learning rate decay factor. Defaults to 0.999.
            batch_size (int): Size of mini-batches for stochastic gradient descent. Defaults to 64.
            early_stopping (bool): Whether to use early stopping to terminate training when
                validation score is not improving. Defaults to True.
            adaptive_lr (bool): Whether to use adaptive learning rate. Defaults to False.
            verbose (int): Level of verbosity. 0 means no information, 1 means convergence
                information, 2 means detailed information. Defaults to 0.
            gamma (str, optional): Kernel coefficient for ‘rbf’, ‘poly’,
                and ‘sigmoid’. Defaults to "scale".
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
        self.gamma = gamma
        self._valid_kernels = ["linear", "polynomial", "rbf"]
        self._valid_gammas = ["scale", "auto"]

        # setting the valid criterions for svc and svr
        if isinstance(self, SVC):
            self._valid_score_metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "confusion_matrix",
            ]
        elif isinstance(self, SVR):
            self._valid_score_metrics = [
                "r_squared",
                "mse",
                "mae",
                "rmse",
                "medae",
                "mape",
                "msle",
                "max_error",
            ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Function responsible for fitting the Decision Tree model.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
        """

    def score(
        self, X: np.ndarray, y: np.ndarray, metric: str = "accuracy", **kwargs
    ) -> Union[np.float64, np.ndarray]:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            metric (string): which metric should be used to assess
                the model's performance. Defaults to Accuracy.

        Returns:
            (np.float32, np.ndarray): the score achieved by the model
                or its confusion matrix.
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """

    def _validate_parameters(self) -> None:
        """
        Validates the parameters of the SVC instance.

        Raises:
            ValueError: If any parameter is invalid.
        """


class SVC(SVMBase):
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
        gamma: str = "scale",
    ) -> None:
        """
        Creates a Support Vector Classifier (SVC) instance.

        Args:
            C (float): Regularization parameter. Defaults to 0.35.
            alpha (float): Learning rate for gradient descent. Defaults to 0.01.
            kernel (str): Kernel type to be used in the algorithm. Valid values are "linear",
                "polynomial", and "rbf". Defaults to "linear".
            degree (int): Degree of the polynomial kernel function ('poly').
                Ignored by other kernels. Defaults to 3.
            max_iter (int): Maximum number of iterations for training. Defaults to 1500.
            tol (float): Tolerance for stopping criteria. Defaults to 1e-4.
            learning_rate (float): Initial learning rate for gradient descent. Defaults to 1e-5.
            decay (float): Learning rate decay factor. Defaults to 0.999.
            batch_size (int): Size of mini-batches for stochastic gradient descent. Defaults to 64.
            early_stopping (bool): Whether to use early stopping to terminate training when
                validation score is not improving. Defaults to True.
            adaptive_lr (bool): Whether to use adaptive learning rate. Defaults to False.
            verbose (int): Level of verbosity. 0 means no information, 1 means convergence
                information, 2 means detailed information. Defaults to 0.
            gamma (str, optional): Kernel coefficient for ‘rbf’, ‘poly’,
                and ‘sigmoid’. Defaults to "scale".
        """
        super().__init__(
            C,
            alpha,
            kernel,
            degree,
            max_iter,
            tol,
            learning_rate,
            decay,
            batch_size,
            early_stopping,
            adaptive_lr,
            verbose,
            gamma,
        )
        self.scaler = StandardScaler()
        self.weights = None
        self.classes_ = None
        self.bias = 0

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
            return linear_kernel(X1, X2)

        if self.kernel == "polynomial":
            return polynomial_kernel(X1, X2, self.degree)

        if self.kernel == "rbf":
            return rbf_kernel(X1, X2, gamma=self.gamma)

    def _check_is_fitted(self) -> None:
        """
        Checks if the model is fitted.

        Raises:
            ValueError: If the model is not trained.
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction.")

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
                "Regularization parameter C must be a positive float.\n"
            ) from error

        try:
            assert isinstance(self.alpha, float) and self.alpha > 0
        except AssertionError as error:
            raise ValueError(
                "Learning rate alpha must be a positive float.\n"
            ) from error

        try:
            assert self.kernel in self._valid_kernels
        except AssertionError as error:
            raise ValueError(
                "Kernel must be one of 'linear', 'polynomial', or 'rbf'.\n"
            ) from error

        try:
            assert self.gamma in self._valid_gammas
        except AssertionError as error:
            raise ValueError("Gamma must be 'auto' or 'scale'.\n") from error

        try:
            assert isinstance(self.degree, int) and self.degree > 0
        except AssertionError as error:
            raise ValueError("Degree must be a positive integer.\n") from error

        try:
            assert isinstance(self.max_iter, int) and self.max_iter > 0
        except AssertionError as error:
            raise ValueError(
                "Maximum number of iterations must be a positive integer.\n"
            ) from error

        try:
            assert isinstance(self.tol, float) and self.tol > 0
        except AssertionError as error:
            raise ValueError("Tolerance must be a positive float.\n") from error

        try:
            assert isinstance(self.learning_rate, float) and self.learning_rate > 0
        except AssertionError as error:
            raise ValueError("Learning rate must be a positive float.\n") from error

        try:
            assert isinstance(self.decay, float) and 0 < self.decay <= 1
        except AssertionError as error:
            raise ValueError("Decay must be a float between 0 and 1.\n") from error

        try:
            assert isinstance(self.batch_size, int) and self.batch_size > 0
        except AssertionError as error:
            raise ValueError("Batch size must be a positive integer.\n") from error

        try:
            assert isinstance(self.early_stopping, bool)
        except AssertionError as error:
            raise ValueError("Early stopping must be a boolean.\n") from error

        try:
            assert isinstance(self.adaptive_lr, bool)
        except AssertionError as error:
            raise ValueError("Adaptive learning rate must be a boolean.\n") from error

        try:
            assert isinstance(self.weights, (type(None), np.ndarray))
        except AssertionError as error:
            raise ValueError("Weights must be None or a numpy array.\n") from error

        try:
            assert isinstance(self.classes_, (type(None), np.ndarray))
        except AssertionError as error:
            raise ValueError("Classes must be None or a numpy array.\n") from error

        try:
            assert isinstance(self.bias, (int, float))
        except AssertionError as error:
            raise ValueError("Bias must be an integer or float.\n") from error

        try:
            assert isinstance(self.scaler, StandardScaler)
        except AssertionError as error:
            raise ValueError(
                "Scaler must be an instance of StandardScaler.\n"
            ) from error

        try:
            assert isinstance(self._valid_score_metrics, list) and all(
                isinstance(metric, str) for metric in self._valid_score_metrics
            )
        except AssertionError as error:
            raise ValueError("Valid metrics must be a list of strings.\n") from error

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the SVC model using mini-batch gradient descent.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """
        self._validate_parameters()

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

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
        self._check_is_fitted()

        X = convert_array_numpy(X)

        X = self.scaler.transform(X)
        linear_output = np.dot(X, self.weights) - self.bias
        predictions = np.sign(linear_output)

        if len(self.classes_) == 2:
            return np.where(predictions == -1, self.classes_[0], self.classes_[1])

        return predictions

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
        labels_cm: List = None,
        normalize_cm: bool = False,
    ) -> Union[np.float64, np.ndarray]:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            metric (string): which metric should be used to assess
                the model's performance. Defaults to Accuracy.
            labels_cm (str, optional): which labels should be used to calculate
                the confusion matrix. If other metric is selected, then this
                parameter will be ignore. Defaults to None.
            normalize_cm (bool, optional): whether the confusion matrix should be
                normalized ('all', 'pred', 'true') or not. If other metric is selected,
                then this parameter will be ignore. Defaults to False.

        Returns:
            (np.float32, np.ndarray): the score achieved by the model
                or its confusion matrix.
        """
        self._check_is_fitted()

        try:
            assert metric in self._valid_score_metrics
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_score_metrics}.\n"
            ) from error

        y_hat = self.predict(X)

        if metric == "accuracy":
            return accuracy(y, y_hat)

        if metric == "precision":
            return precision(y, y_hat)

        if metric == "recall":
            return recall(y, y_hat)

        if metric == "f1_score":
            return f1_score(y, y_hat)

        if metric == "confusion_matrix":
            return confusion_matrix(y, y_hat, labels_cm, normalize_cm)


class SVR(SVMBase):
    """
    Base class for Support Vector Regression (SVR).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        degree: int = 3,
        gamma: str = "scale",
    ) -> None:
        """
        Initializes SVR with default parameters.

        Args:
            kernel (str, optional): Kernel type to be used in the algorithm. Defaults to "rbf".
            C (float, optional): Regularization parameter. Defaults to 1.0.
            epsilon (float, optional): Epsilon parameter in the epsilon-SVR model. Defaults to 0.1.
            degree (int, optional): Degree of the polynomial kernel.
                Ignored by other kernels. Defaults to 3.
            gamma (str, optional): Kernel coefficient for ‘rbf’, ‘poly’,
                and ‘sigmoid’. Defaults to "scale".
        """
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
        )
        self.C = C
        self.epsilon = epsilon
        self.X_ = None
        self.y_ = None
        self.alphas_ = None
        self.b_ = None
        self.K_ = None

    def _validate_parameters(self) -> None:
        """
        Validates the parameters passed during initialization.
        """
        if not isinstance(self.C, (int, float)) or self.C <= 0:
            raise ValueError("C must be a positive number.\n")

        if self.kernel not in self._valid_kernels:
            raise ValueError("Kernel must be one of 'linear', 'poly', or 'rbf'.\n")

        if not isinstance(self.epsilon, (int, float)) or self.epsilon < 0:
            raise ValueError("Epsilon must be a non-negative number.\n")

        if not isinstance(self.degree, int) or self.degree <= 0:
            raise ValueError("Degree must be a positive integer.\n")

        if not isinstance(self.gamma, (str, float)) or (
            isinstance(self.gamma, str) and self.gamma not in self._valid_gammas
        ):
            raise ValueError("Gamma must be 'scale', 'auto', or a positive float.\n")

        if isinstance(self.gamma, float) and self.gamma <= 0:
            raise ValueError("Gamma must be a positive float.\n")

    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Computes the kernel between two sets of data points.

        Args:
            X1 (np.ndarray): First set of data points.
            X2 (np.ndarray): Second set of data points.

        Returns:
            np.ndarray: Kernel matrix.
        """
        if self.kernel == "linear":
            return linear_kernel(X1, X2)

        if self.kernel == "poly":
            return polynomial_kernel(X1, X2, degree=self.degree)

        if self.kernel == "rbf":
            return rbf_kernel(X1, X2, gamma=self.gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the SVR model to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self._validate_parameters()

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        self.X_ = X
        self.y_ = y

        # Compute the kernel matrix
        self.K_ = self._kernel_function(X, X)

        n_samples = X.shape[0]
        K = self.K_

        # Create P matrix
        P_top = np.hstack((K, -K))
        P_bottom = np.hstack((-K, K))
        P = np.vstack((P_top, P_bottom))

        # Ensure P is positive semi-definite
        P = P.astype(np.float64) + 1e-8 * np.eye(2 * n_samples)

        # Create q vector
        q = np.hstack([self.epsilon - y, self.epsilon + y])

        # Create G matrix and h vector for inequality constraints
        G_std = np.vstack((-np.eye(2 * n_samples), np.eye(2 * n_samples)))
        h_std = np.hstack((np.zeros(2 * n_samples), self.C * np.ones(2 * n_samples)))

        # Create A matrix and b vector for equality constraint
        A = np.hstack((np.ones(n_samples), -np.ones(n_samples))).reshape(1, -1)
        b = np.array([0.0])

        P = matrix(P)
        q = matrix(q)
        G = matrix(G_std)
        h = matrix(h_std)
        A = matrix(A)
        b = matrix(b)

        # Solve QP problem
        solvers.options["show_progress"] = False  # Suppress output
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract alphas
        z = np.array(solution["x"]).flatten()
        alpha = z[:n_samples]
        alpha_star = z[n_samples:]
        self.alphas_ = alpha - alpha_star

        # Compute bias term
        f = self.K_ @ self.alphas_
        idx = np.where((alpha > 1e-5) & (alpha < self.C - 1e-5))[0]
        idx_star = np.where((alpha_star > 1e-5) & (alpha_star < self.C - 1e-5))[0]

        b_list = []

        for i in idx:
            b_i = y[i] - f[i] - self.epsilon
            b_list.append(b_i)

        for i in idx_star:
            b_i = y[i] - f[i] + self.epsilon
            b_list.append(b_i)

        if b_list:
            self.b_ = np.mean(b_list)
        else:
            self.b_ = 0.0  # Default to zero if no support vectors found

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given feature matrix.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.X_ is None or self.alphas_ is None:
            raise ValueError(
                "The model has not been trained yet. Please call the fit method first."
            )

        X = convert_array_numpy(X)
        K_pred = self._kernel_function(self.X_, X)
        predictions = (self.alphas_ @ K_pred) + self.b_
        return predictions

    def score(
        self, X: np.ndarray, y: np.ndarray, metric: str = "r_squared"
    ) -> np.float64:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            metric (string): which metric should be used to assess
                the model's performance. Defaults to R Squared.

        Returns:
            np.float32: the score achieved by the model.
        """
        try:
            assert metric in self._valid_score_metrics
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_score_metrics}.\n"
            ) from error

        y_hat = self.predict(X)

        if metric == "r_squared":
            return r_squared(y, y_hat)

        if metric == "mse":
            return mean_squared_error(y, y_hat)

        if metric == "mae":
            return mean_absolute_error(y, y_hat)

        if metric == "rmse":
            return root_mean_squared_error(y, y_hat)

        if metric == "medae":
            return median_absolute_error(y, y_hat)

        if metric == "mape":
            return mean_absolute_percentage_error(y, y_hat)

        if metric == "msle":
            return mean_squared_logarithmic_error(y, y_hat)

        if metric == "max_error":
            return max_error(y, y_hat)
