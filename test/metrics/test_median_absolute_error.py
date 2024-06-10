from sklearn.metrics import median_absolute_error as SkMedAE
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.metrics import median_absolute_error
from test.utils import generate_regression_dataset, repeat
import unittest
import numpy as np


class Test_MedianAbsoluteError(unittest.TestCase):
    """
    Unittest class created to test the Median Absolute Error metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the Median Absolute Error and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=10, n_targets=1)

        sklr = SkLinearRegression()

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkMedAE(y, sklr_prediction)
        score = median_absolute_error(y, sklr_prediction, derivative=False)

        assert np.abs(score - sklr_score) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
