from sklearn.metrics.pairwise import euclidean_distances as SkED
from scratchml.distances import euclidean
from numpy.testing import assert_equal, assert_almost_equal
from test.utils import repeat
import unittest
import numpy as np


class Test_Euclidean(unittest.TestCase):
    """
    Unittest class created to test the Euclidean distance metric.
    """

    @repeat(10)
    def test1(self):
        """
        Test the Euclidean metric implementation on random values and
        then compares it to the Scikit-Learn implementation.
        """
        X = np.random.rand(100, 200)
        y = np.random.rand(300, 200)

        sk_distances = SkED(X, y)
        distances = euclidean(X, y)

        assert_almost_equal(sk_distances, distances)
        assert_equal(type(sk_distances), type(distances))
        assert_equal(sk_distances.shape, distances.shape)

    @repeat(10)
    def test2(self):
        """
        Test the Euclidean metric implementation on random values and
        then compares it to the Scikit-Learn implementation.
        """
        X = np.random.rand(1000, 200)
        y = np.random.rand(3000, 200)

        sk_distances = SkED(X, y)
        distances = euclidean(X, y)

        assert_almost_equal(sk_distances, distances)
        assert_equal(type(sk_distances), type(distances))
        assert_equal(sk_distances.shape, distances.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
