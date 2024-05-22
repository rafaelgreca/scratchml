import numpy as np
from scratchml.losses import mse, r_squared
from scratchml.utils import convert_array_numpy
    
class LinearRegression(object):

    def __init__(
        self,
        learning_rate: float,
        tol: float,
        n_jobs: int = None
    ) -> None:
        self.n_jobs = n_jobs
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in = None
        self.lr = learning_rate
        self.tol = tol
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        self.n_features_in_ = X.shape[1]
        X = convert_array_numpy(X)
        y = convert_array_numpy(y)
        
        self.intercept_ = 0.0
        self.coef_ = np.zeros(X.shape[1])
        last_losses = np.zeros(X.shape[1]) + np.inf

        while True:
            y_hat = self.predict(X)

            loss = mse(y, y_hat, derivative=True)
            
            derivative_coef = (np.matmul(X.T, loss)) / y.shape[0]
            derivative_intercept = (np.sum(loss)) / y.shape[0]

            self.coef_ = self.coef_ - (self.lr * derivative_coef)
            self.intercept_ = self.intercept_ - (self.lr * derivative_intercept)

            if np.max(np.abs(last_losses)) < self.tol:
                break
            
            last_losses = derivative_coef
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        return np.matmul(X, self.coef_) + self.intercept_
    
    def score(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.float32:
        y_hat = self.predict(X)
        return r_squared(y, y_hat)