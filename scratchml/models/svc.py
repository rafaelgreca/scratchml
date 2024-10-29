from scratchml.utils import split_data_into_batches
from scratchml.metrics import mean_squared_error, accuracy, f1_score, confusion_matrix
from scratchml.regularizations import l2
from scratchml.scalers import StandardScaler
import numpy as np

class SVC:
    """
    Support Vector Classifier (SVC) with options for linear, polynomial, and RBF kernels.
    
    Parameters
    ----------
    C : float, optional, default=0.5
        Regularization parameter. The strength of the regularization is inversely proportional to C. 
    alpha : float, optional, default=0.01
        Regularization term for weight updates.
    kernel : str, optional, default="linear"
        Kernel type: 'linear', 'polynomial', or 'rbf'.
    degree : int, optional, default=3
        Degree of the polynomial kernel.
    max_iter : int, optional, default=1000
        Maximum number of iterations for training.
    tol : float, optional, default=1e-4
        Tolerance for stopping criterion.
    learning_rate : float, optional, default=1e-3
        Initial learning rate.
    decay : float, optional, default=0.995
        Learning rate decay factor.
    batch_size : int, optional, default=16
        Number of samples per batch in mini-batch gradient descent.
    early_stopping : bool, optional, default=True
        Stop training early if tolerance threshold is reached.
    adaptive_lr : bool, optional, default=True
        If True, scales learning rate based on gradient norms.
    """
    
    def __init__(
        self,
        C=0.5,
        alpha=0.01,
        kernel="linear",
        degree=3,
        max_iter=1000,
        tol=1e-4,
        learning_rate=1e-3,
        decay=0.995,
        batch_size=16,
        early_stopping=True,
        adaptive_lr=True,
    ):
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
        self.weights = None
        self.bias = 0
        self.scaler = StandardScaler()  # Add scaler for data normalization

    def _apply_kernel(self, X1, X2):
        """Applies the selected kernel function."""
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "polynomial":
            return (1 + np.dot(X1, X2.T)) ** self.degree
        elif self.kernel == "rbf":
            gamma = 1 / X1.shape[1]
            return np.exp(-gamma * np.square(np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)))

    def fit(self, X, y):
        """
        Trains the SVC model using mini-batch gradient descent.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape (n_samples,)
            Target labels (-1 or 1).
        """
        X = self.scaler.fit_transform(X)  # Scale the data
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)

        for iteration in range(self.max_iter):
            lr = self.learning_rate * (self.decay ** iteration)
            avg_update_norm = 0
            for X_batch, y_batch in split_data_into_batches(X, y_, self.batch_size, shuffle=True):
                weight_update = np.zeros_like(self.weights)
                bias_update = 0
                for idx in range(X_batch.shape[0]):
                    instance = X_batch[idx:idx+1]
                    margin = y_batch[idx] * (np.dot(instance.flatten(), self.weights) - self.bias)

                    if margin < 1:
                        # Use hinge loss for gradient update
                        weight_update += -self.C * y_batch[idx] * instance.flatten()
                        bias_update += -self.C * y_batch[idx]

                # Apply L2 regularization to weight update
                weight_update += self.alpha * l2(self.weights)

                # Adaptive learning rate adjustment
                lr_adjusted = lr / (1 + np.linalg.norm(weight_update)) if self.adaptive_lr else lr
                self.weights -= lr_adjusted * weight_update
                self.bias -= lr_adjusted * bias_update
                avg_update_norm += np.linalg.norm(weight_update)

            avg_update_norm /= (X.shape[0] / self.batch_size)
            if self.early_stopping and avg_update_norm < self.tol:
                print(f"Converged after {iteration} iterations.")
                break
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Avg norm of batch updates = {avg_update_norm}")

    def predict(self, X):
        """
        Predicts class labels for input data.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        ndarray
            Predicted labels (-1 or 1).
        """
        X = self.scaler.transform(X)  # Scale the data
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)

    def evaluate(self, X, y):
        """
        Evaluates the model using accuracy, F1 score, and confusion matrix.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to evaluate.
        y : ndarray, shape (n_samples,)
            True labels.

        Returns
        -------
        dict
            Dictionary containing 'accuracy', 'f1_score', and 'confusion_matrix' for the model.
        """
        predictions = self.predict(X)
        return {
            "accuracy": accuracy(y, predictions),
            "f1_score": f1_score(y, predictions),
            "confusion_matrix": confusion_matrix(y, predictions)
        }
