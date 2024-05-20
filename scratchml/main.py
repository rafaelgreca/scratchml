import os
from sklearn.datasets import make_regression
from scratchml.models.linear_regression import LinearRegression

if __name__ == "__main__":
    X, y = make_regression(
        n_samples=10000,
        n_features=1,
        n_targets=1,
        shuffle=True,
        noise=15
    )

    lr = LinearRegression(learning_rate=0.001, tol=1e-06)
    lr.fit(X, y)

    print(lr.coef_)
    print(lr.intercept_)