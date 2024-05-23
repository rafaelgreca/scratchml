import unittest
import numpy as np
from sklearn.metrics import mean_squared_error as SkMSE
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scratchml.metrics import mean_squared_error
from test.utils import generate_regression_dataset, repeat

class Test_MeanSquaredError(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_regression_dataset(
            n_samples=10000,
            n_features=10,
            n_targets=1
        )

        sklr = SkLinearRegression()

        sklr.fit(X, y)

        sklr_prediction = sklr.predict(X)

        sklr_score = SkMSE(y, sklr_prediction, squared=True)
        score = mean_squared_error(y, sklr_prediction, derivative=False)

        assert np.abs(score - sklr_score) < 2
        
if __name__ == "__main__":
    unittest.main(verbosity=2)