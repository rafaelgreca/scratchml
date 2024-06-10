from sklearn.metrics import mean_absolute_percentage_error as SkMAPE
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.metrics import mean_absolute_percentage_error
from test.utils import generate_regression_dataset, repeat
import unittest
import numpy as np


class Test_MeanAbsolutePercentageError(unittest.TestCase):
    """
    Unittest class created to test the Mean Absolute Percentage Error metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the Mean Absolute Percentage Error and then compares it to the
        Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=10, n_targets=1)

        sklr = SkLinearRegression()

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkMAPE(y, sklr_prediction)
        score = mean_absolute_percentage_error(y, sklr_prediction, derivative=False)

        assert np.abs(score - sklr_score) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
