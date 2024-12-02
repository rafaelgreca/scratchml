from scratchml.models.svr import BaseSVR
from scratchml.utils import train_test_split
from sklearn.datasets import make_regression


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

    # creating a linear regression model
    svr = BaseSVR(kernel="linear", C=1.0, epsilon=0.1)

    # fitting the model
    svr.fit(X=X_train, y=y_train)

    # assessing the model's performance
    score = svr.score(X=X_test, y=y_test, metric="r_squared")

    print(f"The model achieved a R² score of {score}.\n")


if __name__ == "__main__":
    example_svr()
