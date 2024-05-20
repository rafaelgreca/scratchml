import unittest
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.models.linear_regression import LinearRegression
from test.utils import generate_regression_dataset, repeat

class Test_LinearRegression(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_regression_dataset()

        sklr = SkLinearRegression()
        lr = LinearRegression(learning_rate=0.001, tol=1e-06)

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=5)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=5)
    
    @repeat(10)
    def test_2(self):
        X, y = generate_regression_dataset(n_features=100)

        sklr = SkLinearRegression()
        lr = LinearRegression(learning_rate=0.001, tol=1e-06)

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=5)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=5)
    
    @repeat(10)
    def test_3(self):
        X, y = generate_regression_dataset(n_features=1000)

        sklr = SkLinearRegression()
        lr = LinearRegression(learning_rate=0.001, tol=1e-06)

        sklr.fit(X, y)
        lr.fit(X, y)

        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert_almost_equal(sklr.coef_, lr.coef_, decimal=5)
        assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=5)

if __name__ == "__main__":
    unittest.main(verbosity=2)