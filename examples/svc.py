from scratchml.models.svc import SVC
from sklearn.datasets import make_classification
from scratchml.utils import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Print evaluation metrics
    print("Test Set Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")


if __name__ == "__main__":
    example_svc()
