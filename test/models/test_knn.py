import unittest
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from scratchml.models.knn import KNNClassifier
from test.utils import repeat, generate_blob_dataset

class Test_KNN(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_blob_dataset(
            n_samples=2000,
            n_features=2
        )

        sk_knn = SkKNN(n_neighbors=5, metric="manhattan")
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
        assert_almost_equal(sk_score, score, decimal=2)
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)
    
    @repeat(10)
    def test_2(self):
        X, y = generate_blob_dataset(
            n_samples=2000,
            n_features=2
        )

        sk_knn = SkKNN(n_neighbors=3, metric="euclidean", weights="distance")
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
        assert_almost_equal(sk_score, score, decimal=2)
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)

    @repeat(10)
    def test_3(self):
        X, y = generate_blob_dataset(
            n_samples=1000,
            n_features=5
        )

        sk_knn = SkKNN(n_neighbors=5)
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
        assert_almost_equal(sk_score, score, decimal=2)
        assert_almost_equal(sk_neighbors, neighbors, decimal=6)
        assert_equal(sk_neighbors[0].shape, neighbors[0].shape)
        assert_equal(sk_neighbors[1].shape, neighbors[1].shape)

if __name__ == "__main__":
    unittest.main(verbosity=2)