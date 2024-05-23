from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from scratchml.models.logistic_regression import LogisticRegression

if __name__ == "__main__":
    X, y = make_classification(n_samples=15000, n_features=10, n_classes=2, n_clusters_per_class=2)

    lr = LogisticRegression(learning_rate=0.1, tol=1e-4)
    lr.fit(X, y)

    print(lr.coef_)
    print(lr.intercept_)
    print(lr.score(X, y))
    print(lr.classes_)

    sklr = SkLogisticRegression(penalty='none', fit_intercept=True, max_iter=1000000, tol=1e-4)
    sklr.fit(X, y)

    print(sklr.coef_)
    print(sklr.intercept_)
    print(sklr.score(X, y))
    print(sklr.classes_)