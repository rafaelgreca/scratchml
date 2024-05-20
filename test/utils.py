from sklearn.datasets import make_regression
from typing import Tuple

def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)

        return callHelper

    return repeatHelper

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