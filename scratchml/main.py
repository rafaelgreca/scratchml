import numpy as np
from sklearn.datasets import make_classification
from scratchml.utils import train_test_split

if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, n_clusters_per_class=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X=X,
        y=y,
        test_size=0.2,
        shuffle=True,
        stratify=False
    )

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    unique, counts = np.unique(y_train, return_counts=True)
    counts = np.asarray((unique, counts)).T
    print(counts)

    unique, counts = np.unique(y_test, return_counts=True)
    counts = np.asarray((unique, counts)).T
    print(counts)