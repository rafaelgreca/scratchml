from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.models.linear_regression import LinearRegression
from test.utils import generate_regression_dataset, repeat
import unittest
import numpy as np


class Test_LinearRegression(unittest.TestCase):
    """
    Unittest class created to test the Linear Regression implementation.
    """

    @repeat(5)
    def test_1(self):
        """
        Test the Linear Regression implementation
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset()

        sklr = SkLinearRegression()
        lr = LinearRegression(learning_rate=0.1, tol=1e-06, verbose=0)

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=5)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=5)
        assert_almost_equal(sklr.score(X, y), lr.score(X, y), decimal=5)

    @repeat(5)
    def test_2(self):
        """
        Test the Linear Regression implementation on a higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_features=50)

        sklr = SkLinearRegression()
        lr = LinearRegression(learning_rate=0.1, tol=1e-06, verbose=0)

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=5)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=5)
        assert_almost_equal(sklr.score(X, y), lr.score(X, y), decimal=5)

    @repeat(5)
    def test_3(self):
        """
        Test the Linear Regression implementation on an even higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_features=100)

        sklr = SkLinearRegression()
        lr = LinearRegression(learning_rate=0.1, tol=1e-06, verbose=0)

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=5)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=5)
        assert_almost_equal(sklr.score(X, y), lr.score(X, y), decimal=5)

    @repeat(5)
    def test_4(self):
        """
        Test the Linear Regression implementation using 'l1' regularization
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset()

        sklr = SkLinearRegression()
        lr = LinearRegression(
            learning_rate=0.1, tol=1e-06, regularization="l1", verbose=0
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=2)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=2)
        assert_almost_equal(sklr.score(X, y), lr.score(X, y), decimal=3)

    @repeat(5)
    def test_5(self):
        """
        Test the Linear Regression implementation using 'l1' regularization
        on a higher dimension dataset and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=20)

        sklr = SkLinearRegression()
        lr = LinearRegression(
            learning_rate=0.1, tol=1e-06, regularization="l1", verbose=0
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=1)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=1)
        assert_almost_equal(sklr.score(X, y), lr.score(X, y), decimal=3)

    @repeat(5)
    def test_6(self):
        """
        Test the Linear Regression implementation using 'l2' regularization
        and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset()

        sklr = SkLinearRegression()
        lr = LinearRegression(
            learning_rate=0.1, tol=1e-06, regularization="l2", verbose=0
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert np.max(np.abs(sklr.coef_ - lr.coef_)) < 3
        assert np.max(np.abs(sklr.intercept_ - lr.intercept_)) < 1
        assert_almost_equal(sklr.score(X, y), lr.score(X, y), decimal=3)

    @repeat(5)
    def test_7(self):
        """
        Test the Linear Regression implementation using 'l2' regularization
        on a higher dimension dataset and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=20)

        sklr = SkLinearRegression()
        lr = LinearRegression(
            learning_rate=0.1, tol=1e-06, regularization="l2", verbose=0
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert np.max(np.abs(sklr.coef_ - lr.coef_)) < 3
        assert np.max(np.abs(sklr.intercept_ - lr.intercept_)) < 1
        assert_almost_equal(sklr.score(X, y), lr.score(X, y), decimal=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
