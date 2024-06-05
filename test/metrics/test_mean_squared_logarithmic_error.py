import unittest
import numpy as np
from sklearn.metrics import mean_squared_log_error as SkMSLE
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.metrics import mean_squared_logarithmic_error
from test.utils import generate_regression_dataset, repeat


class Test_MeanSquaredLogarithmicError(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_regression_dataset(n_samples=10000, n_features=10, n_targets=1)

        sklr = SkLinearRegression()

        # MSLE only accepts positive targets
        y = np.abs(y)

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkMSLE(y, sklr_prediction)
        score = mean_squared_logarithmic_error(y, sklr_prediction, derivative=False)

        assert np.abs(score - sklr_score) < 0.1


if __name__ == "__main__":
    unittest.main(verbosity=2)
