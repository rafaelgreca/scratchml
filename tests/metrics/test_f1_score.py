from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import f1_score as SkF1
from scratchml.metrics import f1_score
from ..utils import generate_classification_dataset, repeat
import unittest
import numpy as np


class Test_F1Score(unittest.TestCase):
    """
    Unittest class created to test the F1-Score metric implementation.
    """

    @repeat(3)
    def test_1(self):
        """
        Test the F1-Score metric on a binary dataset and then compares it to the
        Scikit-Learn implementation.
        """
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

    @repeat(3)
    def test_2(self):
        """
        Test the F1-Score metric on a multi-class dataset and then compares it to the
        Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_features=6, n_samples=7000, n_classes=3
        )

        sklr = SkLogisticRegression(
            penalty=None,
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            random_state=42,
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = SkF1(y, sklr_prediction, average="weighted")
        score = f1_score(y, sklr_prediction, average="weighted")
        assert np.abs(score - sklr_score) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
