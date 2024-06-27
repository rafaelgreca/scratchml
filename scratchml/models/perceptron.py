from ..utils import convert_array_numpy
from ..metrics import accuracy, recall, precision, f1_score, confusion_matrix
from ..regularizations import l1, l2
from ..activations import sigmoid
from typing import Union, List, Tuple
import numpy as np


class Perceptron:
    """
    Creates a class for the Perceptron model.
    """

    def __init__(
        self,
        penalty: str = None,
        lr: float = 0.001,
        alpha: float = 0.0001,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 0.001,
        verbose: int = 0,
        n_jobs: int = None,
    ) -> None:
        """
        Creates a Perceptron instance.

        Args:
            penalty (str, optional): which regularization penalty
                to use, can be 'l1', 'l2' or None. Defaults to None.
            lr (float, optional): the learning rate value. Defaults to 0.001.
            alpha (float, optional): the regularization lambda. Defaults to 0.0001.
            fit_intercept (bool, optional): whether to fit the intercept
                (bias) or not. Defaults to True.
            max_iter (int, optional): the maximum number of iterations
                during the model training. -1 means that no maximum
                iterations is used. Defaults to 1000.
            tol (float, optional): the tolerance of the difference between two sequential
                lossed, which is used as a stopping criteria. Defaults to 0.001.
            verbose (int, optional): how much information should be printed.
                Should be 0, 1, or 2. Defaults to 2.
            n_jobs (int, optional): the number of jobs to be used.
                -1 means that all CPUs are used to train the model. Defaults to None.
        """
        self.penalty = penalty
        self.lr = lr
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self._valid_metrics = [
            "accuracy",
            "recall",
            "precision",
            "f1_score",
            "confusion_matrix",
        ]
        self._valid_regularizations = ["l1", "l2", None]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Function responsible for fitting the linear model.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
        """
        self._validate_parameters()

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        X = convert_array_numpy(X)
        y = convert_array_numpy(y).reshape(-1, 1)

        if len(self.classes_) == 2:
            self.intercept_ = np.zeros((1,), dtype=np.float64)
            self.coef_ = np.zeros((X.shape[1], 1), dtype=np.float64)
            self.coef_, self.intercept_ = self._fitting_model(
                X, y, self.coef_, self.intercept_
            )
        else:
            raise RuntimeError(
                "Perceptron can only be use for binary classification!\n"
            )

    def _fitting_model(
        self, X: np.ndarray, y: np.ndarray, coefs: np.ndarray, intercept: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auxiliary function that is used to fit the model.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            coefs (np.ndarray): the coefficients array.
            intercept (np.ndarray): the intercept array.

        Returns:
            coefs, intercept (np.ndarray, np.ndarray): the
                new coefficient and intercept arrays, respectively.
        """
        epoch = 1
        last_losses = np.zeros((1, self.n_features_in_)) + np.inf

        while True:
            # making the prediction
            y_hat = self.predict(X)

            loss = y - y_hat
            derivative_coef = np.matmul(X.T, loss)
            derivative_intercept = np.sum(np.multiply(self.lr, loss))

            # applying the regularization to the loss function
            if self.penalty is not None:
                reg_coef = np.zeros((1, self.n_features_in_))
                reg_intercept = np.zeros((1, self.n_features_in_))

                if self.penalty == "l1":
                    reg_coef = l1(coefs, reg_lambda=self.alpha, derivative=True)

                    if self.fit_intercept:
                        reg_intercept = l1(
                            intercept, reg_lambda=self.alpha, derivative=True
                        )

                elif self.penalty == "l2":
                    reg_coef = l2(coefs, reg_lambda=self.alpha, derivative=True)

                    if self.fit_intercept:
                        reg_intercept = l2(
                            intercept, reg_lambda=self.alpha, derivative=True
                        )

                derivative_coef += reg_coef

                if self.fit_intercept:
                    derivative_intercept += reg_intercept

            # updating the coefficients
            coefs += self.lr * derivative_coef

            # updating the bias parameter
            if self.fit_intercept:
                intercept += self.lr * derivative_intercept

            if self.verbose != 0:
                metric_msg = f"Metric (Accuracy): {self.score(X, y)}"

                if self.max_iter != -1:
                    epoch_msg = f"Epoch: {epoch}/{self.max_iter}"
                else:
                    epoch_msg = f"Epoch: {epoch}"

                if self.verbose == 1:
                    if epoch % 20 == 0:
                        print(f"{epoch_msg}\t\t{metric_msg}\n")
                elif self.verbose == 2:
                    print(f"{epoch_msg}\t\t{metric_msg}\n")

            # stopping criteria
            if np.max(np.abs(last_losses - derivative_coef)) < self.tol:
                break

            if self.max_iter != -1:
                if epoch > self.max_iter:
                    break

            last_losses = derivative_coef
            epoch += 1

        return coefs, intercept

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        set (also called features).

        Args:
            X (np.ndarray): the features.
            threshold (float, optional): the threshold of the prediction.
                Defaults to 0.5.

        Returns:
            np.ndarray: the predicted classes.
        """
        y_hat = np.matmul(X, self.coef_) + self.intercept_
        y_hat = sigmoid(y_hat)  # activation function
        y_hat = (y_hat > threshold).astype(int)
        return y_hat

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
        labels_cm: List = None,
        normalize_cm: bool = False,
    ) -> Union[np.float32, np.ndarray]:
        """
        Calculates the score of the model on a given set for a
        determined metric.

        Args:
            X (np.ndarray): the features.
            y (np.ndarray): the targets of the features.
            threshold (float): the threshold of the prediction. Defaults to 0.5.
            metric (str, optional): which metric to use. Defaults to "r_squared".
            labels_cm (str, optional): which labels should be used to calculate
                the confusion matrix. If other metric is selected, then this
                parameter will be ignore. Defaults to None.
            normalize_cm (bool, optional): whether the confusion matrix should be
                normalized ('all', 'pred', 'true') or not. If other metric is selected,
                then this parameter will be ignore. Defaults to False.

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

    def _validate_parameters(self) -> None:
        """
        Auxiliary function used to validate the values of the parameters
        passed during the initialization.
        """
        # validating the tol value
        try:
            assert self.tol > 0
        except AssertionError as error:
            raise ValueError("The 'tol' must be bigger than zero.\n") from error

        # validating the learning rate value
        try:
            assert self.lr > 0
        except AssertionError as error:
            raise ValueError("The 'lr' must be bigger than zero.\n") from error

        # validating the fit intercept parameter
        try:
            assert isinstance(self.fit_intercept, bool)
        except AssertionError as error:
            raise ValueError(
                "The 'fit_intercept' must be a boolean value.\n"
            ) from error

        # validating the alpha value
        try:
            assert self.alpha > 0
        except AssertionError as error:
            raise ValueError("The 'alpha' must be bigger than zero.\n") from error

        # validating the regularization parameter
        try:
            if self.penalty is not None:
                assert self.penalty in self._valid_regularizations
        except AssertionError as error:
            raise ValueError(
                f"The 'penalty' must be {self._valid_regularizations}.\n"
            ) from error

        # validating the max_iters value
        if self.max_iter < -1 or self.max_iter == 0:
            raise ValueError(
                "Invalid value for 'max_iter'. Must be -1 or >= 1.\n"
            ) from error

        # validating the n_jobs value
        if self.n_jobs is not None:
            try:
                if self.n_jobs < 0:
                    assert self.n_jobs == -1
                else:
                    assert self.n_jobs > 0
            except AssertionError as error:
                raise ValueError(
                    "If not None, 'n_jobs' must be equal to -1 or higher than 0.\n"
                ) from error

        # validating the verbose value
        try:
            assert self.verbose in [0, 1, 2]
        except AssertionError as error:
            raise ValueError(
                "Indalid value for 'verbose'. Must be 0, 1, or 2.\n"
            ) from error
