import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import f1_score as SkF1
from scratchml.metrics import f1_score
from test.utils import generate_classification_dataset, repeat


class Test_Precision(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=2
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = SkF1(y, sklr_prediction)
        acc_score = f1_score(y, sklr_prediction)

        assert np.abs(acc_score - sklr_score) < 0.05

    @repeat(10)
    def test_2(self):
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=5
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = SkF1(y, sklr_prediction, average="micro")
        score = f1_score(y, sklr_prediction, average="micro")
        assert np.abs(score - sklr_score) < 0.05

        sklr_score = SkF1(y, sklr_prediction, average="macro")
        score = f1_score(y, sklr_prediction, average="macro")

        assert np.abs(score - sklr_score) < 0.05

        sklr_score = SkF1(y, sklr_prediction, average="weighted")
        score = f1_score(y, sklr_prediction, average="weighted")

        assert np.abs(score - sklr_score) < 0.05


if __name__ == "__main__":
    unittest.main(verbosity=2)
