from sklearn.metrics.pairwise import manhattan_distances as SkMD
from scratchml.distances import manhattan
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import numpy as np


class Test_Manhattan(unittest.TestCase):
    """
    Unittest class created to test the Manhattan distance metric.
    """

    @repeat(10)
    def test1(self):
        """
        Test the Manhattan metric implementation on random values and
        then compares it to the Scikit-Learn implementation.
        """
        X = np.random.rand(100, 200)
        y = np.random.rand(300, 200)

        sk_distances = SkMD(X, y)
        distances = manhattan(X, y)

        assert_almost_equal(sk_distances, distances)
        assert_equal(type(sk_distances), type(distances))
        assert_equal(sk_distances.shape, distances.shape)

    @repeat(10)
    def test2(self):
        """
        Test the Manhattan metric implementation on random values and
        then compares it to the Scikit-Learn implementation.
        """
        X = np.random.rand(1000, 200)
        y = np.random.rand(3000, 200)

        sk_distances = SkMD(X, y)
        distances = manhattan(X, y)

        assert_almost_equal(sk_distances, distances)
        assert_equal(type(sk_distances), type(distances))
        assert_equal(sk_distances.shape, distances.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
