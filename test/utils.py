from sklearn.datasets import make_regression, make_classification
from typing import Tuple

def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)

        return callHelper

    return repeatHelper

def generate_classification_dataset(
    n_samples: int = 10000,
    n_features: int = 5,
    n_classes: int = 2,
    n_clusters_per_class: int = 1
) -> Tuple:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0
    )
    return X, y
    
def generate_regression_dataset(
    n_samples: int = 10000,
    n_features: int = 1,
    n_targets: int = 1,
    shuffle: bool = True,
    noise: int = 20,
    coef: bool = False
) -> Tuple:
    if coef:
        X, y, coef = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_targets,
            shuffle=shuffle,
            noise=noise,
            coef=coef
        )
        return X, y, coef
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_targets,
            shuffle=shuffle,
            noise=noise,
            coef=coef
        )
        return X, y