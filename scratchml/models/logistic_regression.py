import numpy as np
from scratchml.losses import binary_cross_entropy
from scratchml.utils import convert_array_numpy
from scratchml.activations import sigmoid
from scratchml.metrics import accuracy

class LogisticRegression(object):

    def __init__(
        self,
        learning_rate: float,
        tol: float,
        n_jobs: int = None,
        max_iters: int = -1,
        loss_function: str = "bce"
    ) -> None:
        """
        Creates a Logistic Regression instance.

        Args:
            learning_rate (float): the learning rate used to train the model.
            tol (float): the tolerance of the difference between two sequential
                lossed, which is used as a stopping criteria.
            max_iters (int, optional): the maximum number of iterations
                during the model training. -1 means that no maximum
                iterations is used. Defaults to -1.
            loss_function (str, optional): the loss function to be used.
                Defaults to "mse".
            n_jobs (int, optional): the number of jobs to be used.
                -1 means that all CPUs are used to train the model. Defaults to None.
        """
        self.n_jobs = n_jobs
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.lr = learning_rate
        self.tol = tol
        self.max_iters = max_iters
        self.loss_function = loss_function
        self._valid_loss_functions = ["bce"]
        self._valid_metrics = ["accuracy"]
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)
        
        # validating the max_iters value
        if self.max_iters < -1 or self.max_iters == 0:
            return ValueError("Invalid value for 'max_iters'. Must be -1 or >= 1.\n")
        
        # validating the loss_function value
        try:
            assert self.loss_function in self._valid_loss_functions
        except AssertionError:
            return ValueError(
                f"Invalid value for 'loss_function'. Must be {self._valid_loss_functions}.\n"
            )
        
        self.intercept_ = np.zeros((1, ), dtype=np.float64)
        self.coef_ = np.zeros((1, X.shape[1]), dtype=np.float64)
        last_losses = np.zeros(X.shape[1]) + np.inf

        if self.max_iters == -1:
            count = -99
        else:
            count = 1

        while True:
            # making the prediction
            y_hat = np.matmul(X, self.coef_.T) + self.intercept_        
            y_hat = np.squeeze(sigmoid(y_hat))
            
            # calculating the loss according to the chosen
            # loss function
            if self.loss_function == "bce":
                loss = binary_cross_entropy(y, y_hat, derivative=True)
                derivative_coef = (np.matmul(X.T, loss)) / y.shape[0]
                derivative_intercept = (np.sum(loss)) / y.shape[0]

            # updating the coefficients
            self.coef_ -= (self.lr * derivative_coef)
            self.intercept_ -= (self.lr * derivative_intercept)

            # stopping criteria
            if (np.max(np.abs(last_losses)) < self.tol) or \
                (count >= self.max_iters):
                break
            
            last_losses = derivative_coef

            if self.max_iters != -1:
                count += 1
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        set (also called features).

        Args:
            X (np.ndarray): the features.
            threshold (float): the threshold of the prediction. Defaults to 0.5.

        Returns:
            np.ndarray: the predicted classes.
        """
        y_hat = np.matmul(X, self.coef_.T) + self.intercept_
        y_hat = np.squeeze(sigmoid(y_hat))
        y_hat = (y_hat > threshold).astype(int)
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
        metric: str = "accuracy"
    ) -> np.float32:
        """
        Calculates the score of the model on a given set for a
        determined metric.

        Args:
            X (np.ndarray): the features.
            y (np.ndarray): the targets of the features.
            threshold (float): the threshold of the prediction. Defaults to 0.5.
            metric (str, optional): which metric to use. Defaults to "r_squared".

        Returns:
            np.float32: the score achieved by the model.
        """
        try:
            assert metric in self._valid_metrics
        except AssertionError:
            return f"Invalid value for 'metric'. Must be {self._valid_metrics}.\n"
        
        y_hat = self.predict(X, threshold)

        if metric == "accuracy":
            return accuracy(y, y_hat)