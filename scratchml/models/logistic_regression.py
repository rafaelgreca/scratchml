import numpy as np
from scratchml.models.losses import binary_cross_entropy
from scratchml.utils import convert_array_numpy
from scratchml.models.activations import sigmoid
from scratchml.metrics import accuracy

class LogisticRegression(object):

    def __init__(
        self,
        learning_rate: float,
        tol: float,
        n_jobs: int = None
    ) -> None:
        self.n_jobs = n_jobs
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.lr = learning_rate
        self.tol = tol
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)
        
        self.intercept_ = np.zeros((1, ), dtype=np.float64)
        self.coef_ = np.zeros((1, X.shape[1]), dtype=np.float64)
        last_losses = np.zeros(X.shape[1]) + np.inf

        while True:
            y_hat = np.matmul(X, self.coef_.T) + self.intercept_        
            y_hat = np.squeeze(sigmoid(y_hat))
            
            loss = binary_cross_entropy(y, y_hat, derivative=True)

            derivative_coef = (np.matmul(X.T, loss)) / y.shape[0]
            derivative_intercept = (np.sum(loss)) / y.shape[0]

            self.coef_ -= (self.lr * derivative_coef)
            self.intercept_ -= (self.lr * derivative_intercept)

            if np.max(np.abs(last_losses)) < self.tol:
                break
            
            last_losses = derivative_coef
    
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        y_hat = np.matmul(X, self.coef_.T) + self.intercept_
        y_hat = sigmoid(y_hat)
        y_hat = (y_hat > threshold).astype(int)
        return y_hat

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        y_hat = np.matmul(X, self.coef_.T) + self.intercept_
        y_hat = sigmoid(y_hat)
        return y_hat

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> np.float32:
        y_hat = self.predict(X, threshold)
        return accuracy(y, y_hat)