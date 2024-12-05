from sklearn.metrics import mean_squared_error as SkMSE
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.metrics import mean_squared_error, root_mean_squared_error
from ..utils import generate_regression_dataset, repeat
import unittest
import warnings
import numpy as np


class Test_MeanSquaredError(unittest.TestCase):
    """
    Unittest class created to test the Mean Squared Error and the
    Root Mean Squared Error metrics implementation.
    """

    def setUp(self):
        warnings.simplefilter("ignore", category=FutureWarning)

    @repeat(3)
    def test_1(self):
        """
        Test the Mean Squared Error and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=10, n_targets=1)

        sklr = SkLinearRegression()

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkMSE(y, sklr_prediction, squared=True)
        score = mean_squared_error(y, sklr_prediction, derivative=False)

        assert np.abs(score - sklr_score) < 0.1

    @repeat(3)
    def test_2(self):
        """
        Test the Root Mean Squared Error and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=10, n_targets=1)

        sklr = SkLinearRegression()

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkMSE(y, sklr_prediction, squared=False)
        score = root_mean_squared_error(y, sklr_prediction, derivative=False)

        assert np.abs(score - sklr_score) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
