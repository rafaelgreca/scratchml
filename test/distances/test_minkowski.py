import unittest
import numpy as np
from sklearn.metrics import DistanceMetric
from scratchml.distances import minkowski
from numpy.testing import assert_equal, assert_almost_equal
from test.utils import repeat

class Test_Chebyshev(unittest.TestCase):
    def test1(self):
        p = 2.0
        X = [[0, 1], [1, 1]]
        test = [[0, 0]]

        X = np.asarray(X)
        test = np.asarray(test)

        dist = DistanceMetric.get_metric("minkowski", p=p)
        sk_distances = dist.pairwise(X, X)
        distances = minkowski(X, X, p)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)

        sk_distances = dist.pairwise(X, test)
        distances = minkowski(X, test, p)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)

    @repeat(10)
    def test2(self):
        p = 1.0
        X = np.random.rand(100, 200)
        y = np.random.rand(300, 200)

        dist = DistanceMetric.get_metric("minkowski", p=p)
        sk_distances = dist.pairwise(X, y)
        distances = minkowski(X, y, p)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)
    
    @repeat(10)
    def test3(self):
        p = 10.0
        X = np.random.rand(1000, 200)
        y = np.random.rand(3000, 200)

        dist = DistanceMetric.get_metric("minkowski", p=p)
        sk_distances = dist.pairwise(X, y)
        distances = minkowski(X, y, p)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)
    
if __name__ == "__main__":
    unittest.main(verbosity=2)