import unittest
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances as SkMD
from scratchml.distances import manhattan
from numpy.testing import assert_equal, assert_almost_equal
from test.utils import repeat


class Test_Manhattan(unittest.TestCase):
    def test1(self):
        X = [[0, 1], [1, 1]]
        test = [[0, 0]]

        X = np.asarray(X)
        test = np.asarray(test)

        sk_distances = SkMD(X, X)
        distances = manhattan(X, X)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)

        sk_distances = SkMD(X, test)
        distances = manhattan(X, test)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)

    @repeat(10)
    def test2(self):
        X = np.random.rand(100, 200)
        y = np.random.rand(300, 200)

        sk_distances = SkMD(X, y)
        distances = manhattan(X, y)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)

    @repeat(10)
    def test3(self):
        X = np.random.rand(1000, 200)
        y = np.random.rand(3000, 200)

        sk_distances = SkMD(X, y)
        distances = manhattan(X, y)

        assert_almost_equal(sk_distances, distances)
        assert type(sk_distances) == type(distances)
        assert_equal(sk_distances.shape, distances.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
