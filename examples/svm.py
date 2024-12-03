from sklearn.datasets import make_classification, make_regression
from scratchml.models.svm import SVC, SVR
from scratchml.utils import train_test_split
from scratchml.metrics import (
    accuracy,
    f1_score,
    precision,
    recall,
)
import numpy as np


def example_svc():
    """
    Example of how to use the optimized SVC model with enhanced metrics,
    polynomial kernel, and cross-validation for high precision.
    """
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_classes=2,
        n_informative=15,
        n_redundant=5,
        class_sep=1.8,
        random_state=42,
    )
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for SVM compatibility

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the SVC model with optimal parameters
    model = SVC(
        C=0.4,
        alpha=0.015,
        kernel="linear",
        degree=4,
        max_iter=1000,
        tol=1e-5,
        learning_rate=5e-4,
        decay=0.995,
        batch_size=16,
        adaptive_lr=True,
        gamma="scale",
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy_score = accuracy(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision_score = precision(y_test, y_pred)
    recall_score = recall(y_test, y_pred)

    # Print evaluation metrics
    print("Test Set Results:")
    print(f"Accuracy: {accuracy_score * 100:.2f}%")
    print(f"Precision: {precision_score * 100:.2f}%")
    print(f"Recall: {recall_score * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}\n")


def example_svr() -> None:
    """
    Practical example of how to use the Support Vector Regression (SVR) model.
    """
    # generating a dataset for the regression task
    X, y = make_regression(
        n_samples=2000, n_features=5, n_targets=1, shuffle=True, noise=30
    )

    # splitting the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=True, stratify=False
    )

    # creating a SVR model
    svr = SVR(kernel="linear", C=1.0, epsilon=0.1)

    # fitting the model
    svr.fit(X=X_train, y=y_train)

    # assessing the model's performance
    score = svr.score(X=X_test, y=y_test, metric="r_squared")

    print(f"The model achieved a R² score of {score}.\n")


if __name__ == "__main__":
    example_svc()
    example_svr()
