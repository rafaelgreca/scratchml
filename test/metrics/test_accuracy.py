from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score
from ...scratchml.metrics import accuracy
from ..utils import generate_classification_dataset, repeat
import unittest
import numpy as np


class Test_Accuracy(unittest.TestCase):
    """
    Unittest class created to test the Accuracy metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the accuracy metric on a binary dataset and then compares it to the Scikit-Learn
        implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=2
        )

        sklr = SkLogisticRegression(
            penalty="none", fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = accuracy_score(y, sklr_prediction)
        acc_score = accuracy(y, sklr_prediction)

        assert np.abs(acc_score - sklr_score) < 0.1

    @repeat(10)
    def test_2(self):
        """
        Test the accuracy metric on a multi-class dataset and then compares it to the Scikit-Learn
        implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=10
        )

        sklr = SkLogisticRegression(
            penalty="none", fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = accuracy_score(y, sklr_prediction)
        acc_score = accuracy(y, sklr_prediction)

        assert np.abs(acc_score - sklr_score) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
