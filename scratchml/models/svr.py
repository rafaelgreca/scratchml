from abc import ABC
import numpy as np
from ..utils import convert_array_numpy
from ..metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r_squared,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    max_error,
)
from ..kernels import linear_kernel, polynomial_kernel, rbf_kernel


class BaseSVR(ABC):
    """
    Base class for Support Vector Regression (SVR).
    """

    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1, degree=3, gamma="scale"):
        """
        Initializes SVR with default parameters.

        Args:
            kernel (str, optional): Kernel type to be used in the algorithm. Defaults to "rbf".
            C (float, optional): Regularization parameter. Defaults to 1.0.
            epsilon (float, optional): Epsilon parameter in the epsilon-SVR model. Defaults to 0.1.
            degree (int, optional): Degree of the polynomial kernel. Ignored by other kernels. Defaults to 3.
            gamma (str, optional): Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’. Defaults to "scale".
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.degree = degree
        self.gamma = gamma
        self.X_ = None
        self.y_ = None
        self.alphas_ = None
        self.b_ = None
        self.K_ = None

    def _kernel_function(self, X1, X2):
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
        elif self.kernel == "poly":
            return polynomial_kernel(X1, X2, degree=self.degree)
        elif self.kernel == "rbf":
            return rbf_kernel(X1, X2, gamma=self.gamma)
        else:
            raise ValueError("Unknown kernel specified")

    def fit(self, X, y):
        """
        Fits the SVR model to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        self.X_ = X
        self.y_ = y

        # Set gamma if necessary
        if self.gamma == 'scale':
            self.gamma_ = 1 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            self.gamma_ = 1 / X.shape[1]
        else:
            self.gamma_ = float(self.gamma)

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

        # Convert to cvxopt matrices
        from cvxopt import matrix, solvers
        P = matrix(P)
        q = matrix(q)
        G = matrix(G_std)
        h = matrix(h_std)
        A = matrix(A)
        b = matrix(b)

        # Solve QP problem
        solvers.options['show_progress'] = False  # Suppress output
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract alphas
        z = np.array(solution['x']).flatten()
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

    def predict(self, X):
        """
        Predicts target values for the given feature matrix.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.X_ is None or self.alphas_ is None:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

        X = convert_array_numpy(X)
        K_pred = self._kernel_function(self.X_, X)
        predictions = (self.alphas_ @ K_pred) + self.b_
        return predictions

    def score(self, X, y, metric="r_squared"):
        """
        Evaluates the model on a test dataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True target values.
            metric (str, optional): Evaluation metric. Defaults to "r_squared".

        Returns:
            np.float64: Score based on the chosen metric.
        """
        y_hat = self.predict(X)

        if metric == "r_squared":
            return r_squared(y, y_hat)
        elif metric == "mse":
            return mean_squared_error(y, y_hat)
        elif metric == "mae":
            return mean_absolute_error(y, y_hat)
        elif metric == "rmse":
            return root_mean_squared_error(y, y_hat)
        elif metric == "medae":
            return median_absolute_error(y, y_hat)
        elif metric == "mape":
            return mean_absolute_percentage_error(y, y_hat)
        elif metric == "msle":
            return mean_squared_logarithmic_error(y, y_hat)
        elif metric == "max_error":
            return max_error(y, y_hat)
        else:
            raise ValueError(f"Unknown metric: {metric}")

