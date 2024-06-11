from numpy.testing import assert_almost_equal
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import roc_auc_score as SkRAC
from ...scratchml.metrics import roc_auc_score
from ..utils import generate_classification_dataset, repeat
import unittest


class Test_ROCAUCScore(unittest.TestCase):
    """
    Unittest class created to test the ROC AUC Score metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the ROC AUC Score on a binary dataset and then compares it to the Scikit-Learn
        implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=2
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict_proba(X)[:, 1]
        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="macro")
        acc_score = roc_auc_score(y, sklr_prediction, average="macro")

        assert_almost_equal(sklr_score, acc_score)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="micro")
        acc_score = roc_auc_score(y, sklr_prediction, average="micro")

        assert_almost_equal(sklr_score, acc_score)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="weighted")
        acc_score = roc_auc_score(y, sklr_prediction, average="weighted")

        assert_almost_equal(sklr_score, acc_score)

    @repeat(10)
    def test_2(self):
        """
        Test the ROC AUC Score on a multi-class dataset and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=6
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict_proba(X)
        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="macro")
        acc_score = roc_auc_score(y, sklr_prediction, average="macro")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="micro")
        acc_score = roc_auc_score(y, sklr_prediction, average="micro")

        assert_almost_equal(sklr_score, acc_score, decimal=2)

        sklr_score = SkRAC(y, sklr_prediction, multi_class="ovr", average="weighted")
        acc_score = roc_auc_score(y, sklr_prediction, average="weighted")

        assert_almost_equal(sklr_score, acc_score, decimal=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
