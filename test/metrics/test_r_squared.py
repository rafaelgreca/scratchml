from sklearn.metrics import r2_score as SkR2
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.metrics import r_squared
from test.utils import generate_regression_dataset, repeat
import unittest
import numpy as np


class Test_RSquared(unittest.TestCase):
    """
    Unittest class created to test the R Squared metric implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the R Squared and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=10000, n_features=10, n_targets=1)

        sklr = SkLinearRegression()

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkR2(y, sklr_prediction)
        score = r_squared(y, sklr_prediction)

        assert np.abs(score - sklr_score) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
