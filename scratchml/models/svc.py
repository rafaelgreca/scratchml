from scratchml.utils import split_data_into_batches
from scratchml.metrics import mean_squared_error, accuracy, f1_score, confusion_matrix
from scratchml.regularizations import l2
from scratchml.scalers import StandardScaler
import numpy as np

class SVC:
    """
    Support Vector Classifier (SVC) with options for linear, polynomial, and RBF kernels.
    """
    
    def __init__(
        self,
        C=0.35,
        alpha=0.01,
        kernel="linear",
        degree=3,
        gamma=None,
        coef0=1,
        max_iter=3000,
        tol=1e-4,
        learning_rate=1e-5,  # Lower initial learning rate
        decay=0.999,  # Slower decay
        batch_size=64,
        early_stopping=True,
        adaptive_lr=False,
    ):
        # Parameter validation
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        
        self.C = C
        self.alpha = alpha
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.adaptive_lr = adaptive_lr
        self.weights = None
        self.classes_ = None
        self.bias = 0
        self.scaler = StandardScaler()

    def _apply_kernel(self, X1, X2):
        """Applies the selected kernel function."""
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "polynomial":
            return (self.coef0 + np.dot(X1, X2.T)) ** self.degree
        elif self.kernel == "rbf":
            gamma = self.gamma if self.gamma is not None else 1 / X1.shape[1]
            return np.exp(-gamma * np.square(np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)))


    def fit(self, X, y):
        """
        Trains the SVC model using mini-batch gradient descent.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape (n_samples,)
            Target labels (e.g., 0 or 1 for binary classification).
        """
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            y_ = np.where(y == self.classes_[0], -1, 1)
        else:
            y_ = y

        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        no_improvement_count = 0  # Track convergence across iterations
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
                        weight_update += -self.C * y_batch[idx] * instance.flatten()
                        bias_update += -self.C * y_batch[idx]

                weight_update += self.alpha * l2(self.weights)

                lr_adjusted = lr / (1 + np.linalg.norm(weight_update)) if self.adaptive_lr else lr
                self.weights -= lr_adjusted * weight_update
                self.bias -= lr_adjusted * bias_update
                avg_update_norm += np.linalg.norm(weight_update)

            avg_update_norm /= (X.shape[0] / self.batch_size)
            if self.early_stopping and avg_update_norm < self.tol:
                no_improvement_count += 1
                if no_improvement_count > 10:  # Terminate if no improvement for 10 iterations
                    print(f"Converged after {iteration} iterations.")
                    break
            else:
                no_improvement_count = 0

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Avg norm of batch updates = {avg_update_norm}")

        print(f"Final weight norm: {np.linalg.norm(self.weights)}")
        print(f"Final bias: {self.bias}")

    
    def predict(self, X):
        """
        Predicts class labels for input data.
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

    def evaluate(self, X, y):
        """
        Evaluates the model using accuracy, F1 score, and confusion matrix.
        """
        predictions = self.predict(X)
        return {
            "accuracy": accuracy(y, predictions),
            "f1_score": f1_score(y, predictions),
            "confusion_matrix": confusion_matrix(y, predictions)
        }
    
    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test data.
        y : ndarray, shape (n_samples,)
            True labels.
        
        Returns
        -------
        score : float
            Mean accuracy of the model on the test data.
        """
        predictions = self.predict(X)
        return accuracy(y, predictions)
