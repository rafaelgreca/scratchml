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
from ..utils import convert_array_numpy
from ..regularizations import l1, l2
from typing import Union
import numpy as np


class LinearRegression:
    """
    Creates a class for the Linear Regression model.
    """

    def __init__(
        self,
        learning_rate: float,
        tol: float,
        max_iters: int = -1,
        loss_function: str = "mse",
        regularization: Union[None, str] = None,
        n_jobs: int = None,
        verbose: int = 2,
    ) -> None:
        """
        Creates a Linear Regression instance.

        Args:
            learning_rate (float): the learning rate used to train the model.
            tol (float): the tolerance of the difference between two sequential
                lossed, which is used as a stopping criteria.
            max_iters (int, optional): the maximum number of iterations
                during the model training. -1 means that no maximum
                iterations is used. Defaults to -1.
            loss_function (str, optional): the loss function to be used.
                Defaults to "mse".
            regularization (str | None, optional): the regularization function
                that will be used in the model training. None means that
                no regularization function will be used. Defaults to None.
            n_jobs (int, optional): the number of jobs to be used.
                -1 means that all CPUs are used to train the model. Defaults to None.
            verbose (int, optional): how much information should be printed.
                Should be 0, 1, or 2. Defaults to 2.
        """
        self.n_jobs = n_jobs
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.lr = learning_rate
        self.tol = tol
        self.max_iters = max_iters
        self.loss_function = loss_function
        self.regularization = regularization
        self.verbose = verbose
        self._valid_loss_functions = ["mse", "mae"]
        self._valid_metrics = [
            "r_squared",
            "mse",
            "mae",
            "rmse",
            "medae",
            "mape",
            "msle",
            "max_error",
        ]
        self._valid_regularizations = ["l1", "l2", None]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Function responsible for training the Linear Regression model.

        Args:
            X (np.ndarray): the features.
            y (np.ndarray): the targets of the features.
        """
        self.n_features_in_ = X.shape[1]
        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        # validating the max_iters value
        if self.max_iters < -1 or self.max_iters == 0:
            raise ValueError("Invalid value for 'max_iters'. Must be -1 or >= 1.\n")

        # validating the loss_function value
        try:
            assert self.loss_function in self._valid_loss_functions
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'loss_function'. Must be {self._valid_loss_functions}.\n"
            ) from error

        # validating the regularization function value
        try:
            assert self.regularization in self._valid_regularizations
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'regularization'. Must be {self._valid_regularizations}.\n"
            ) from error

        # validating the verbose value
        try:
            assert self.verbose in [0, 1, 2]
        except AssertionError as error:
            raise ValueError(
                "Indalid value for 'verbose'. Must be 0, 1, or 2.\n"
            ) from error

        self.intercept_ = 0.0
        self.coef_ = np.zeros(X.shape[1])
        last_losses = np.zeros((1, X.shape[1])) + np.inf
        epoch = 1

        # training loop
        while True:
            # making the prediction
            y_hat = self.predict(X)
            loss = 0.0

            # calculating the loss according to the chosen
            # loss function
            if self.loss_function == "mse":
                loss = mean_squared_error(y, y_hat, derivative=True)
            elif self.loss_function == "mae":
                loss = mean_absolute_error(y, y_hat, derivative=True)

            derivative_coef = (np.matmul(X.T, loss)) / y.shape[0]
            derivative_intercept = (np.sum(loss)) / y.shape[0]

            # applying the regularization to the loss function
            if self.regularization is not None:
                reg_coef, reg_intercept = 0.0, 0.0

                if self.regularization == "l1":
                    reg_coef = l1(self.coef_, derivative=True)
                    reg_intercept = l1(self.intercept_, derivative=True)

                elif self.regularization == "l2":
                    reg_coef = l2(self.coef_, derivative=True)
                    reg_intercept = l2(self.intercept_, derivative=True)

                derivative_coef += reg_coef
                derivative_intercept += reg_intercept

            # updating the coefficients
            self.coef_ -= self.lr * derivative_coef
            self.intercept_ -= self.lr * derivative_intercept

            if self.verbose != 0:
                loss_msg = f"Loss ({self.loss_function}): {loss}"
                metric_msg = f"Metric (R²): {self.score(X, y)}"

                if self.max_iters != -1:
                    epoch_msg = f"Epoch: {epoch}/{self.max_iters}"
                else:
                    epoch_msg = f"Epoch: {epoch}"

                if self.verbose == 1:
                    if epoch % 20 == 0:
                        print(f"{epoch_msg}\t\t{loss_msg}\t\t{metric_msg}\n")
                elif self.verbose == 2:
                    print(f"{epoch_msg}\t\t{loss_msg}\t\t{metric_msg}\n")

            # stopping criterias
            if np.max(np.abs(last_losses - derivative_coef)) < self.tol:
                break

            if self.max_iters != -1:
                if epoch >= self.max_iters:
                    break

            last_losses = derivative_coef
            epoch += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the targets of a given
        set (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted targets.
        """
        return np.matmul(X, self.coef_) + self.intercept_

    def score(
        self, X: np.ndarray, y: np.ndarray, metric: str = "r_squared"
    ) -> np.float32:
        """
        Calculates the score of the model on a given set for a
        determined metric.

        Args:
            X (np.ndarray): the features.
            y (np.ndarray): the targets of the features.
            metric (str, optional): which metric to use. Defaults to "r_squared".

        Returns:
            np.float32: the score achieved by the model.
        """
        try:
            assert metric in self._valid_metrics
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_metrics}.\n"
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
