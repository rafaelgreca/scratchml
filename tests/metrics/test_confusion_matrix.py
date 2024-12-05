from numpy.testing import assert_equal
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import confusion_matrix as SkCM
from scratchml.metrics import confusion_matrix
from ..utils import generate_classification_dataset, repeat
import unittest


class Test_ConfusionMatrix(unittest.TestCase):
    """
    Unittest class created to test the Confusion Matrix metric implementation.
    """

    @repeat(3)
    def test_1(self):
        """
        Test the confusion matrix metric on a binary dataset and then compares it to the
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
        sklr_score = SkCM(y, sklr_prediction)
        score = confusion_matrix(y, sklr_prediction)

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, normalize="all")
        score = confusion_matrix(y, sklr_prediction, normalize="all")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, normalize="true")
        score = confusion_matrix(y, sklr_prediction, normalize="true")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, normalize="pred")
        score = confusion_matrix(y, sklr_prediction, normalize="pred")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

    @repeat(3)
    def test_2(self):
        """
        Test the confusion matrix metric on a multi-class dataset and then compares
        it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=10
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)
        sklr_score = SkCM(y, sklr_prediction)
        score = confusion_matrix(y, sklr_prediction)

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, normalize="all")
        score = confusion_matrix(y, sklr_prediction, normalize="all")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, normalize="true")
        score = confusion_matrix(y, sklr_prediction, normalize="true")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, normalize="pred")
        score = confusion_matrix(y, sklr_prediction, normalize="pred")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

    @repeat(3)
    def test_3(self):
        """
        Test the confusion matrix metric on a binary dataset using only a set of labels
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_features=10, n_samples=10000, n_classes=10
        )

        sklr = SkLogisticRegression(
            penalty=None, fit_intercept=True, max_iter=1000000, tol=1e-4
        )

        sklr.fit(X, y)

        labels = [0, 1, 2, 3, 4]

        sklr_prediction = sklr.predict(X)
        sklr_score = SkCM(y, sklr_prediction, labels=labels)
        score = confusion_matrix(y, sklr_prediction, labels=labels)

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, labels=labels, normalize="all")
        score = confusion_matrix(y, sklr_prediction, labels=labels, normalize="all")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, labels=labels, normalize="true")
        score = confusion_matrix(y, sklr_prediction, labels=labels, normalize="true")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))

        sklr_score = SkCM(y, sklr_prediction, labels=labels, normalize="pred")
        score = confusion_matrix(y, sklr_prediction, labels=labels, normalize="pred")

        assert_equal(sklr_score, score)
        assert_equal(sklr_score.shape, score.shape)
        assert_equal(type(sklr_score), type(score))


if __name__ == "__main__":
    unittest.main(verbosity=2)
