from numpy.testing import assert_allclose, assert_equal
from sklearn.linear_model import Perceptron as SkPerceptron
from scratchml.models.perceptron import Perceptron
from ..utils import generate_classification_dataset, generate_blob_dataset, repeat
import unittest
import math
import numpy as np


class Test_Perceptron(unittest.TestCase):
    """
    Unittest class created to test the Perceptron implementation.
    """

    @repeat(3)
    def test_1(self):
        """
        Test the Perceptron implementation and then compares it to
        the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=5000, n_features=2, n_classes=2, n_clusters_per_class=1
        )

        perceptron = Perceptron(
            penalty=None,
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty=None,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_2(self):
        """
        Test the Perceptron implementation with samples with higher dimensions
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=10000, n_features=10, n_classes=2, n_clusters_per_class=1
        )

        perceptron = Perceptron(
            penalty=None,
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty=None,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_3(self):
        """
        Test the Perceptron implementation and then compares it to
        the Scikit-Learn implementation. (using l1 regularization)
        """
        X, y = generate_classification_dataset(
            n_samples=5000, n_features=2, n_classes=2, n_clusters_per_class=1
        )

        perceptron = Perceptron(
            penalty="l1",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l1",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_4(self):
        """
        Test the Perceptron implementation with samples with higher dimensions
        and then compares it to the Scikit-Learn implementation. (using l1 regularization)
        """
        X, y = generate_classification_dataset(
            n_samples=10000, n_features=10, n_classes=2, n_clusters_per_class=1
        )

        perceptron = Perceptron(
            penalty="l1",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l1",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_5(self):
        """
        Test the Perceptron implementation and then compares it to
        the Scikit-Learn implementation. (using l2 regularization)
        """
        X, y = generate_classification_dataset(
            n_samples=5000, n_features=2, n_classes=2, n_clusters_per_class=1
        )

        perceptron = Perceptron(
            penalty="l2",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_6(self):
        """
        Test the Perceptron implementation with samples with higher dimensions
        and then compares it to the Scikit-Learn implementation. (using l2 regularization)
        """
        X, y = generate_classification_dataset(
            n_samples=10000, n_features=10, n_classes=2, n_clusters_per_class=1
        )

        perceptron = Perceptron(
            penalty="l2",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_7(self):
        """
        Test the Perceptron implementation with blob samples and then compares it to
        the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(n_samples=5000, n_features=2, shuffle=True)

        perceptron = Perceptron(
            penalty=None,
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty=None,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_8(self):
        """
        Test the Perceptron implementation with blob samples with higher dimensions
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(n_samples=10000, n_features=10, shuffle=True)

        perceptron = Perceptron(
            penalty=None,
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty=None,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_9(self):
        """
        Test the Perceptron implementation with blob samples and then compares it to
        the Scikit-Learn implementation. (using l1 regularization)
        """
        X, y = generate_blob_dataset(n_samples=5000, n_features=2, shuffle=True)

        perceptron = Perceptron(
            penalty="l1",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l1",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_10(self):
        """
        Test the Perceptron implementation with blob samples with higher dimensions
        and then compares it to the Scikit-Learn implementation. (using l1 regularization)
        """
        X, y = generate_blob_dataset(n_samples=10000, n_features=10, shuffle=True)

        perceptron = Perceptron(
            penalty="l1",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l1",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_11(self):
        """
        Test the Perceptron implementation with blob samples and then compares it to
        the Scikit-Learn implementation. (using l2 regularization)
        """
        X, y = generate_blob_dataset(n_samples=5000, n_features=2, shuffle=True)

        perceptron = Perceptron(
            penalty="l2",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)

    @repeat(3)
    def test_12(self):
        """
        Test the Perceptron implementation with blob samples with higher dimensions
        and then compares it to the Scikit-Learn implementation. (using l2 regularization)
        """
        X, y = generate_blob_dataset(n_samples=10000, n_features=10, shuffle=True)

        perceptron = Perceptron(
            penalty="l2",
            lr=0.001,
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )
        skperceptron = SkPerceptron(
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            verbose=0,
            n_jobs=None,
        )

        skperceptron.fit(X, y)
        perceptron.fit(X, y)

        predict_skp = skperceptron.predict(X)
        predict_p = np.squeeze(perceptron.predict(X))

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(skperceptron.coef_.shape, perceptron.coef_.reshape(1, -1).shape)
        assert_equal(skperceptron.intercept_.shape, perceptron.intercept_.shape)
        assert_equal(skperceptron.n_features_in_, perceptron.n_features_in_)
        # assert_equal(skperceptron.classes_, perceptron.classes_)
        assert_equal(predict_skp.shape, predict_p.shape)
        assert_allclose(predict_skp, predict_p, atol=atol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
