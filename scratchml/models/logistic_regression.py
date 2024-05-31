import numpy as np
from scratchml.losses import binary_cross_entropy
from scratchml.utils import convert_array_numpy
from scratchml.activations import sigmoid, softmax
from scratchml.metrics import (
    accuracy,
    recall,
    precision,
    f1_score,
    confusion_matrix
)
from scratchml.regularizations import l1, l2
from typing import Union, List, Tuple

class LogisticRegression(object):

    def __init__(
        self,
        learning_rate: float,
        tol: float,
        fit_intercept: bool = True,
        n_jobs: int = None,
        max_iters: int = -1,
        loss_function: str = "bce",
        regularization: Union[None, str] = None,
        verbose: int = 2
    ) -> None:
        """
        Creates a Logistic Regression instance.

        Args:
            learning_rate (float): the learning rate used to train the model.
            tol (float): the tolerance of the difference between two sequential
                lossed, which is used as a stopping criteria.
            fit_intercept (bool, optional): whether the intercept/bias should 
                be fitted or not. Defaults to True.
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
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.lr = learning_rate
        self.tol = tol
        self.max_iters = max_iters
        self.loss_function = loss_function
        self.regularization = regularization
        self.verbose = verbose
        self._valid_loss_functions = ["bce"]
        self._valid_metrics = [
            "accuracy",
            "recall",
            "precision",
            "f1_score",
            "confusion_matrix"
        ]
        self._valid_regularizations = ["l1", "l2", None]
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Function responsible for fitting the linear model.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
        """
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)
        
        # validating the max_iters value
        if self.max_iters < -1 or self.max_iters == 0:
            raise ValueError("Invalid value for 'max_iters'. Must be -1 or >= 1.\n")
        
        # validating the loss_function value
        try:
            assert self.loss_function in self._valid_loss_functions
        except AssertionError:
            raise ValueError(
                f"Invalid value for 'loss_function'. Must be {self._valid_loss_functions}.\n"
            )

        # validating the regularization function value
        try:
            assert self.regularization in self._valid_regularizations
        except AssertionError:
            raise ValueError(
                f"Invalid value for 'regularization'. Must be {self._valid_regularizations}.\n"
            )
        
        # validating the verbose value
        try:
            assert self.verbose in [0, 1, 2]
        except AssertionError:
            raise ValueError(
                f"Indalid value for 'verbose'. Must be 0, 1, or 2.\n"
            )
        
        # validating the number of unique classes
        try:
            assert len(self.classes_) >= 2
        except AssertionError:
            raise RuntimeError("Only one unique class was found.\n")

        if len(self.classes_) == 2:
            self.intercept_ = np.zeros((1,), dtype=np.float64)
            self.coef_ = np.zeros((1, X.shape[1]), dtype=np.float64)
            self.coef_, self.intercept_ = self._fitting_model(
                X,
                y,
                self.coef_,
                self.intercept_
            )
        else:
            self.intercept_ = np.zeros(len(self.classes_), dtype=np.float64)
            self.coef_ = np.zeros((len(self.classes_), X.shape[1]), dtype=np.float64)

            for i, c in enumerate(self.classes_):
                _y = (y == c).astype(int)
                self.coef_[i], self.intercept_[i] = self._fitting_model(
                    X,
                    _y,
                    self.coef_[i],
                    self.intercept_[i]
                )

    def _fitting_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefs: np.ndarray,
        intercept: np.ndarray
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
        last_losses = np.zeros((1, X.shape[1])) + np.inf

        while True:
            # making the prediction
            y_hat = np.matmul(X, coefs.T) + intercept
            y_hat = np.squeeze(sigmoid(y_hat))
                        
            # calculating the loss according to the chosen
            # loss function
            if self.loss_function == "bce":
                loss = binary_cross_entropy(y, y_hat, derivative=True)
                derivative_coef = (np.matmul(X.T, loss)) / y.shape[0]

                if self.fit_intercept:
                    derivative_intercept = (np.sum(loss)) / y.shape[0]

            # applying the regularization to the loss function
            if self.regularization != None:
                if self.regularization == "l1":
                    reg_coef = l1(coefs, derivative=True)

                    if self.fit_intercept:
                        reg_intercept = l1(intercept, derivative=True)

                elif self.regularization == "l2":
                    reg_coef = l2(coefs, derivative=True)

                    if self.fit_intercept:
                        reg_intercept = l2(intercept, derivative=True)
                
                reg_coef = np.squeeze(reg_coef)

                if self.fit_intercept:
                    reg_intercept = np.squeeze(reg_intercept)

                derivative_coef += reg_coef

                if self.fit_intercept:
                    derivative_intercept += reg_intercept

            # updating the coefficients
            coefs -= (self.lr * derivative_coef)

            if self.fit_intercept:
                intercept -= (self.lr * derivative_intercept)

            if self.verbose != 0:
                loss_msg = f"Loss ({self.loss_function}): {loss}"
                metric_msg = f"Metric (Accuracy): {self.score(X, y)}"

                if self.max_iters != -1:
                    epoch_msg = f"Epoch: {epoch}/{self.max_iters}"
                else:
                    epoch_msg = f"Epoch: {epoch}"
                                    
                if self.verbose == 1:
                    if epoch % 20 == 0:
                        print(f"{epoch_msg}\t\t{loss_msg}\t\t{metric_msg}\n")
                elif self.verbose == 2:
                    print(f"{epoch_msg}\t\t{loss_msg}\t\t{metric_msg}\n")

            # stopping criteria
            if (np.max(np.abs(last_losses - derivative_coef)) < self.tol):
                break
            
            if self.max_iters != -1:
                if epoch >= self.max_iters:
                    break

            last_losses = derivative_coef
            epoch += 1

        return coefs, intercept
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = None
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        set (also called features).

        Args:
            X (np.ndarray): the features.
            threshold (float, optional): the threshold of the prediction.
                Defaults to None.

        Returns:
            np.ndarray: the predicted classes.
        """
        y_hat = np.matmul(X, self.coef_.T) + self.intercept_
        y_hat = np.squeeze(sigmoid(y_hat))

        if threshold != None:
            y_hat = (y_hat > threshold).astype(int)
        
        if len(self.classes_) > 2:
            y_hat = softmax(y_hat)
            y_hat = np.argmax(y_hat, axis=1)

        return y_hat

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes' probabilities
        of a given set (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted probabilities.
        """
        y_hat = np.matmul(X, self.coef_.T) + self.intercept_
        y_hat = np.squeeze(sigmoid(y_hat))
        return y_hat

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        metric: str = "accuracy",
        labels_cm: List = None,
        normalize_cm: bool = False
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
        except AssertionError:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_metrics}.\n"
            )
        
        y_hat = self.predict(X, threshold)

        if metric == "accuracy":
            return accuracy(y, y_hat)
        elif metric == "precision":
            return precision(y, y_hat)
        elif metric == "recall":
            return recall(y, y_hat)
        elif metric == "f1_score":
            return f1_score(y, y_hat)
        elif metric == "confusion_matrix":
            return confusion_matrix(
                y,
                y_hat,
                labels_cm,
                normalize_cm
            )