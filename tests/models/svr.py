import math
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scratchml.models.svr import BaseSVR
from sklearn.svm import SVR as SkSVR
from ..utils import generate_regression_dataset, repeat


class Test_SVR(unittest.TestCase):
    """
    Unit test class for the custom SVR implementation.
    """

    @repeat(3)
    def test_linear_kernel(self):
        """
        Test the custom SVR with linear kernel against Scikit-Learn's SVR.
        """
        X, y = generate_regression_dataset(n_samples=200, n_features=5)

        custom_svr = BaseSVR(kernel="linear", C=1.0, epsilon=0.1)
        sklearn_svr = SkSVR(kernel="linear", C=1.0, epsilon=0.1)

        custom_svr.fit(X, y)
        sklearn_svr.fit(X, y)

        custom_pred = custom_svr.predict(X)
        sklearn_pred = sklearn_svr.predict(X)

        custom_score = custom_svr.score(X, y)
        sklearn_score = sklearn_svr.score(X, y)

        atol = 1e-1
        assert_allclose(custom_pred, sklearn_pred, atol=atol, rtol=1e-2)
        assert abs(custom_score - sklearn_score) / abs(sklearn_score) < 0.1


    def test_untrained_model_prediction_error(self):
        """
        Ensure an error is raised when predicting with an untrained model.
        """
        svr = BaseSVR(kernel="linear")
        X, _ = generate_regression_dataset(n_samples=10, n_features=2)

        with self.assertRaises(ValueError):
            svr.predict(X)

    @repeat(3)
    def test_custom_kernel_initialization(self):
        """
        Ensure the SVR model initializes correctly with a custom kernel.
        """
        svr = BaseSVR(kernel="poly")
        self.assertEqual(
            svr.kernel,
            "poly",
            "Model should initialize with 'poly' kernel.",
        )

    @repeat(3)
    def test_output_type_and_shape(self):
        """
        Validate that the output type and shape of predictions are correct.
        """
        X, y = generate_regression_dataset(n_samples=200, n_features=5)

        custom_svr = BaseSVR(kernel="linear")
        sklearn_svr = SkSVR(kernel="linear")

        custom_svr.fit(X, y)
        sklearn_svr.fit(X, y)

        custom_pred = custom_svr.predict(X)
        sklearn_pred = sklearn_svr.predict(X)

        self.assertIsInstance(custom_pred, np.ndarray)
        self.assertEqual(custom_pred.shape, sklearn_pred.shape)

    @repeat(3)
    def test_model_score_metrics(self):
        """
        Compare the model scores using different metrics.
        """
        # Generate dataset and ensure non-negative targets
        X, y = generate_regression_dataset(n_samples=200, n_features=5)

        # Make y non-negative for MSLE compatibility
        y = np.abs(y)  # Ensure non-negativity

        custom_svr = BaseSVR(kernel="linear")
        custom_svr.fit(X, y)

        metrics = [
            "r_squared",
            "mse",
            "mae",
            "rmse",
            "medae",
            "mape",
            "msle",  # This is the problematic metric
            "max_error",
        ]

        for metric in metrics:
            score = custom_svr.score(X, y, metric=metric)
            self.assertTrue(
                isinstance(score, (float, np.float64)),
                f"Score for metric {metric} should be a float.",
            )
            self.assertFalse(np.isnan(score), f"Score for metric {metric} is NaN.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
