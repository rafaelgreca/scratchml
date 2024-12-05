from numpy.testing import assert_allclose, assert_equal
from sklearn.cluster import KMeans as SkKMeans
from scratchml.models.kmeans import KMeans
from ..utils import generate_classification_dataset, generate_blob_dataset, repeat
import unittest
import math
import numpy as np


class Test_KMeans(unittest.TestCase):
    """
    Unittest class created to test the KMeans implementation.
    """

    @repeat(3)
    def test_1(self):
        """
        Test the KMeans implementation and then compares it to
        the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=5, n_classes=2, n_clusters_per_class=1
        )

        kmeans = KMeans(
            n_init=5, n_clusters=2, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=2, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert (
            np.abs(sk_kmeans.inertia_ - kmeans.inertia_) / np.abs(sk_kmeans.inertia_)
            < 0.01
        )
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.01

    @repeat(3)
    def test_2(self):
        """
        Test the KMeans implementation with a higher dimension dataset and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=5000, n_features=7, n_classes=2, n_clusters_per_class=1
        )

        kmeans = KMeans(
            n_init=5, n_clusters=2, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=2, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert (
            np.abs(sk_kmeans.inertia_ - kmeans.inertia_) / np.abs(sk_kmeans.inertia_)
            < 0.01
        )
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.01

    @repeat(3)
    def test_3(self):
        """
        Test the KMeans implementation with a multiclass dataset and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=5, n_classes=5, n_clusters_per_class=1
        )

        kmeans = KMeans(
            n_init=5, n_clusters=5, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=5, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert (
            np.abs(sk_kmeans.inertia_ - kmeans.inertia_) / np.abs(sk_kmeans.inertia_)
            < 0.01
        )
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.01

    @repeat(3)
    def test_4(self):
        """
        Test the KMeans implementation with a higher dimension multiclass dataset and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=5000, n_features=7, n_classes=5, n_clusters_per_class=1
        )

        kmeans = KMeans(
            n_init=5, n_clusters=5, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=5, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert (
            np.abs(sk_kmeans.inertia_ - kmeans.inertia_) / np.abs(sk_kmeans.inertia_)
            < 0.01
        )
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.01

    @repeat(3)
    def test_5(self):
        """
        Test the KMeans implementation with a blob dataset and then compares it to
        the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(n_samples=2000, n_features=5, shuffle=True)

        kmeans = KMeans(
            n_init=5, n_clusters=2, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=2, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert (
            np.abs(sk_kmeans.inertia_ - kmeans.inertia_) / np.abs(sk_kmeans.inertia_)
            < 0.01
        )
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.01

    @repeat(3)
    def test_6(self):
        """
        Test the KMeans implementation with a higher dimension blob dataset and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(n_samples=5000, n_features=7, shuffle=True)

        kmeans = KMeans(
            n_init=5, n_clusters=2, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=2, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert (
            np.abs(sk_kmeans.inertia_ - kmeans.inertia_) / np.abs(sk_kmeans.inertia_)
            < 0.01
        )
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.01

    @repeat(3)
    def test_7(self):
        """
        Test the KMeans implementation with a multiclass blob dataset and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(
            n_samples=2000, n_features=5, centers=5, shuffle=True
        )

        kmeans = KMeans(
            n_init=5, n_clusters=5, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=5, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert (
            np.abs(sk_kmeans.inertia_ - kmeans.inertia_) / np.abs(sk_kmeans.inertia_)
            < 0.14
        )
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.1

    @repeat(3)
    def test_8(self):
        """
        Test the KMeans implementation with a higher dimension blob multiclass dataset and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(
            n_samples=5000, n_features=7, centers=5, shuffle=True
        )

        kmeans = KMeans(
            n_init=5, n_clusters=5, max_iter=300, tol=0.0001, verbose=0, n_jobs=None
        )

        sk_kmeans = SkKMeans(
            n_clusters=5, n_init=5, max_iter=300, tol=0.0001, verbose=0
        )

        sk_kmeans.fit(X, y)
        kmeans.fit(X, y)

        predict_skk = sk_kmeans.predict(X)
        predict_k = kmeans.predict(X)

        score_skk = sk_kmeans.score(X, y)
        score_k = kmeans.score()

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_kmeans.cluster_centers_.shape, kmeans.cluster_centers_.shape)
        assert_allclose(sk_kmeans.cluster_centers_, kmeans.cluster_centers_, atol=atol)
        assert_equal(sk_kmeans.labels_.shape, kmeans.labels_.shape)
        assert_allclose(sk_kmeans.labels_, kmeans.labels_, atol=atol)
        assert_equal(sk_kmeans.n_features_in_, kmeans.n_features_in_)
        assert_equal(predict_skk.shape, predict_k.shape)
        assert_allclose(predict_skk, predict_k, atol=atol)
        assert np.abs(score_skk - score_k) / np.abs(score_skk) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
