from sklearn.metrics import DistanceMetric
from ...scratchml.distances import minkowski
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import numpy as np


class Test_Minkowski(unittest.TestCase):
    """
    Unittest class created to test the Minkowski distance metric.
    """

    @repeat(10)
    def test1(self):
        """
        Test the Minkowski metric implementation on random values and
        then compares it to the Scikit-Learn implementation.
        """
        p = 1.0
        X = np.random.rand(100, 200)
        y = np.random.rand(300, 200)

        dist = DistanceMetric.get_metric("minkowski", p=p)
        sk_distances = dist.pairwise(X, y)
        distances = minkowski(X, y, p)

        assert_almost_equal(sk_distances, distances)
        assert_equal(type(sk_distances), type(distances))
        assert_equal(sk_distances.shape, distances.shape)

    @repeat(10)
    def test2(self):
        """
        Test the Minkowski metric implementation on random values with a
        different value for p and then compares it to the Scikit-Learn implementation.
        """
        p = 10.0
        X = np.random.rand(1000, 200)
        y = np.random.rand(3000, 200)

        dist = DistanceMetric.get_metric("minkowski", p=p)
        sk_distances = dist.pairwise(X, y)
        distances = minkowski(X, y, p)

        assert_almost_equal(sk_distances, distances)
        assert_equal(type(sk_distances), type(distances))
        assert_equal(sk_distances.shape, distances.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
