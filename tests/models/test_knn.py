from sklearn.neighbors import KNeighborsClassifier as SkKNNC
from sklearn.neighbors import KNeighborsRegressor as SkKNNR
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from scratchml.models.knn import KNNClassifier, KNNRegressor
from ..utils import repeat, generate_blob_dataset, generate_regression_dataset
import unittest
import math
import numpy as np


class Test_KNN(unittest.TestCase):
    """
    Unittest class created to test the KNN implementation.
    """

    @repeat(2)
    def test_1(self):
        """
        Test the KNN implementation on a small dataset using the 'manhattan' metric
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(n_samples=1500, n_features=2)

        sk_knn = SkKNNC(n_neighbors=5, metric="manhattan")
        sk_knn.fit(X, y)
        sk_prediction = sk_knn.predict(X)
        sk_neighbors = sk_knn.kneighbors(X, 2, return_distance=True)
        sk_score = sk_knn.score(X, y)

        knn = KNNClassifier(n_neighbors=5, metric="manhattan")
        knn.fit(X, y)
        prediction = knn.predict(X)
        neighbors = knn.kneighbors(X, 2, return_distance=True)
        score = knn.score(X, y)

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_knn.classes_, knn.classes_)
        assert_equal(sk_knn.effective_metric_, knn.effective_metric_)
        assert_equal(sk_knn.n_features_in_, knn.n_features_in_)
        assert_equal(sk_knn.n_samples_fit_, knn.n_samples_fit_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) < 0.05
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)

    @repeat(2)
    def test_2(self):
        """
        Test the KNN implementation on a small dataset using the 'euclidean' metric
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(n_samples=1500, n_features=2)

        sk_knn = SkKNNC(n_neighbors=3, metric="euclidean", weights="distance")
        sk_knn.fit(X, y)
        sk_prediction = sk_knn.predict(X)
        sk_neighbors = sk_knn.kneighbors(X, 2, return_distance=True)
        sk_score = sk_knn.score(X, y)

        knn = KNNClassifier(n_neighbors=3, metric="euclidean", weights="distance")
        knn.fit(X, y)
        prediction = knn.predict(X)
        neighbors = knn.kneighbors(X, 2, return_distance=True)
        score = knn.score(X, y)

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_knn.classes_, knn.classes_)
        assert_equal(sk_knn.effective_metric_, knn.effective_metric_)
        assert_equal(sk_knn.n_features_in_, knn.n_features_in_)
        assert_equal(sk_knn.n_samples_fit_, knn.n_samples_fit_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) < 0.05
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)

    @repeat(2)
    def test_3(self):
        """
        Test the KNN implementation on a small dataset with higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_blob_dataset(n_samples=1000, n_features=4)

        sk_knn = SkKNNC(n_neighbors=5)
        sk_knn.fit(X, y)
        sk_prediction = sk_knn.predict(X)
        sk_neighbors = sk_knn.kneighbors(X, 3, return_distance=True)
        sk_score = sk_knn.score(X, y)

        knn = KNNClassifier(n_neighbors=5)
        knn.fit(X, y)
        prediction = knn.predict(X)
        neighbors = knn.kneighbors(X, 3, return_distance=True)
        score = knn.score(X, y)

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_knn.classes_, knn.classes_)
        assert_equal(sk_knn.effective_metric_, knn.effective_metric_)
        assert_equal(sk_knn.n_features_in_, knn.n_features_in_)
        assert_equal(sk_knn.n_samples_fit_, knn.n_samples_fit_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) < 0.05
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)

    @repeat(2)
    def test_4(self):
        """
        Test the KNN implementation on a small dataset with higher dimension with more neighbors
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=1500, n_features=3)

        sk_knn = SkKNNR(n_neighbors=5)
        sk_knn.fit(X, y)
        sk_prediction = sk_knn.predict(X)
        sk_neighbors = sk_knn.kneighbors(X, 3, return_distance=True)
        sk_score = sk_knn.score(X, y)

        knn = KNNRegressor(n_neighbors=5)
        knn.fit(X, y)
        prediction = knn.predict(X)
        neighbors = knn.kneighbors(X, 3, return_distance=True)
        score = knn.score(X, y)

        atol = math.floor(y.shape[0] * 0.05)

        assert_equal(sk_knn.effective_metric_, knn.effective_metric_)
        assert_equal(sk_knn.n_features_in_, knn.n_features_in_)
        assert_equal(sk_knn.n_samples_fit_, knn.n_samples_fit_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) < 0.05
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)

    @repeat(2)
    def test_5(self):
        """
        Test the KNN implementation on a small dataset with higher dimension
        using the 'chebyshev' metric and weights based on the neighbors distances
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=1500, n_features=4)

        sk_knn = SkKNNR(n_neighbors=5, weights="distance", metric="chebyshev")
        sk_knn.fit(X, y)
        sk_prediction = sk_knn.predict(X)
        sk_neighbors = sk_knn.kneighbors(X, 3, return_distance=True)
        sk_score = sk_knn.score(X, y)

        knn = KNNRegressor(n_neighbors=5, weights="distance", metric="chebyshev")
        knn.fit(X, y)
        prediction = knn.predict(X)
        neighbors = knn.kneighbors(X, 3, return_distance=True)
        score = knn.score(X, y)

        atol = math.floor(y.shape[0] * 0.13)

        assert_equal(sk_knn.effective_metric_, knn.effective_metric_)
        assert_equal(sk_knn.n_features_in_, knn.n_features_in_)
        assert_equal(sk_knn.n_samples_fit_, knn.n_samples_fit_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) < 0.13
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
