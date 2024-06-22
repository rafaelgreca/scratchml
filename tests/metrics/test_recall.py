from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import recall_score
from scratchml.metrics import recall
from ..utils import generate_classification_dataset, repeat
import unittest
import numpy as np


class Test_Recall(unittest.TestCase):
    """
    Unittest class created to test the Recall metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the Recall on a binary dataset and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=2
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = recall_score(y, sklr_prediction)
        acc_score = recall(y, sklr_prediction)

        assert np.abs(acc_score - sklr_score) < 0.05

    @repeat(10)
    def test_2(self):
        """
        Test the Recall on a multi-class dataset and then compares it to the
        Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=5
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = recall_score(y, sklr_prediction, average="micro")
        score = recall(y, sklr_prediction, average="micro")
        assert np.abs(score - sklr_score) < 0.05

        sklr_score = recall_score(y, sklr_prediction, average="macro")
        score = recall(y, sklr_prediction, average="macro")

        assert np.abs(score - sklr_score) < 0.05

        sklr_score = recall_score(y, sklr_prediction, average="weighted")
        score = recall(y, sklr_prediction, average="weighted")

        assert np.abs(score - sklr_score) < 0.05


if __name__ == "__main__":
    unittest.main(verbosity=2)
