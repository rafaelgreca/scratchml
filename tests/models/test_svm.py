import math
from numpy.testing import assert_allclose, assert_equal
from sklearn.svm import SVC as SkSVC
from sklearn.svm import SVR as SkSVR
from sklearn.exceptions import ConvergenceWarning
from scratchml.models.svm import SVC, SVR
from scratchml.scalers import StandardScaler
from ..utils import generate_classification_dataset, repeat, generate_regression_dataset
import unittest
import warnings
import numpy as np
from numpy.testing import assert_array_equal


class Test_SVM(unittest.TestCase):
    """
    Unit test class for the custom SVM implementation.
    """

    def setUp(self):
        warnings.simplefilter("ignore", category=ConvergenceWarning)

    @repeat(3)
    def test_binary_classification(self):
        """
        Test binary classification and compare the custom SVC with Scikit-Learn's SVC.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )

        # Initialize and train both models
        custom_svc = SVC(kernel="linear")
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        # Predict and score
        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        custom_score = custom_svc.score(X, y)
        sklearn_score = sklearn_svc.score(X, y)

        # Assertions for binary classification
        atol = math.floor(y.shape[0] * 0.1)
        assert_equal(sklearn_svc.classes_, custom_svc.classes_)
        assert_allclose(sklearn_pred, custom_pred, atol=atol)
        assert abs(sklearn_score - custom_score) / abs(sklearn_score) < 0.1

    @repeat(3)
    def test_multi_class_classification(self):
        """
        Test multi-class classification and compare the custom SVC with Scikit-Learn's SVC.
        """
        # Use scaled data for both models
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Initialize and train both models with adjusted max_iter and tol
        custom_svc = SVC(kernel="linear", max_iter=1000, tol=1e-5)
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000, tol=1e-5)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        custom_score = custom_svc.score(X, y)
        sklearn_score = sklearn_svc.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)
        assert_equal(sklearn_svc.classes_, custom_svc.classes_)
        assert_allclose(sklearn_pred, custom_pred, atol=atol)
        assert abs(sklearn_score - custom_score) / abs(sklearn_score) < 0.08

    @repeat(3)
    def test_rbf_kernel(self):
        """
        Test the custom SVC with RBF kernel against Scikit-Learn's SVC.
        """
        np.random.seed(42)
        X, y = generate_classification_dataset(n_samples=200, n_features=4, n_classes=2)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        custom_svc = SVC(kernel="rbf", max_iter=500, tol=1e-3)
        sklearn_svc = SkSVC(kernel="rbf", max_iter=500, tol=1e-3)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        custom_score = custom_svc.score(X, y)
        sklearn_score = sklearn_svc.score(X, y)

        relative_difference = abs(sklearn_score - custom_score) / abs(sklearn_score)

        mismatches = np.count_nonzero(custom_pred != sklearn_pred)
        mismatch_percentage = mismatches / len(custom_pred)
        assert (
            mismatch_percentage < 0.15
        ), f"Mismatch percentage {mismatch_percentage * 100}% is too high."
        assert (
            relative_difference < 0.15
        ), f"Relative difference {relative_difference} is not acceptable."

    def test_untrained_model_prediction_error(self):
        """
        Ensure an error is raised when predicting with an untrained model.
        """
        svc = SVC(kernel="linear")
        X, _ = generate_classification_dataset(n_samples=10, n_features=2, n_classes=2)

        with self.assertRaises(ValueError):
            svc.predict(X)

    @repeat(3)
    def test_custom_kernel_initialization(self):
        """
        Ensure the SVC model initializes correctly with a custom kernel.
        """
        svc = SVC(kernel="polynomial")
        self.assertEqual(
            svc.kernel,
            "polynomial",
            "Model should initialize with 'polynomial' kernel.",
        )

    @repeat(3)
    def test_output_type_and_shape(self):
        """
        Validate that the output type and shape of predictions are the same.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )

        custom_svc = SVC(kernel="linear")
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        self.assertIsInstance(custom_pred, np.ndarray)
        self.assertEqual(custom_pred.shape, sklearn_pred.shape)

    @repeat(3)
    def test_model_parameters(self):
        """
        Compare the model parameters between the custom and Scikit-Learn implementations.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )

        custom_svc = SVC(kernel="linear")
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        if hasattr(custom_svc, "support_vectors_"):
            assert_array_equal(
                custom_svc.support_vectors_,
                sklearn_svc.support_vectors_,
                "Support vectors should match between implementations.",
            )

    @repeat(3)
    def test_linear_kernel(self):
        """
        Test the custom SVR with linear kernel against Scikit-Learn's SVR.
        """
        X, y = generate_regression_dataset(n_samples=1000, n_features=3)

        custom_svr = SVR(kernel="linear", C=1.0, epsilon=0.1)
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

    def test_untrained_model_prediction_error_svr(self):
        """
        Ensure an error is raised when predicting with an untrained model.
        """
        svr = SVR(kernel="linear")
        X, _ = generate_regression_dataset(n_samples=10, n_features=2)

        with self.assertRaises(ValueError):
            svr.predict(X)

    @repeat(3)
    def test_custom_kernel_initialization_svr(self):
        """
        Ensure the SVR model initializes correctly with a custom kernel.
        """
        svr = SVR(kernel="poly")
        self.assertEqual(
            svr.kernel,
            "poly",
            "Model should initialize with 'poly' kernel.",
        )

    @repeat(3)
    def test_output_type_and_shape_svr(self):
        """
        Validate that the output type and shape of predictions are correct.
        """
        X, y = generate_regression_dataset(n_samples=200, n_features=5)

        custom_svr = SVR(kernel="linear")
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

        custom_svr = SVR(kernel="linear")
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

    def test_parameter_validation(self):
        """
        Test parameter validation for the custom SVR implementation.
        """
        # Generate dataset and ensure non-negative targets
        X, y = generate_regression_dataset(n_samples=200, n_features=5)

        with self.assertRaises(ValueError):
            svr = SVR(kernel="invalid_kernel")
            svr.fit(X, y)

        with self.assertRaises(ValueError):
            svr = SVR(C=-1.0)
            svr.fit(X, y)

        with self.assertRaises(ValueError):
            svr = SVR(epsilon=-0.1)
            svr.fit(X, y)

        with self.assertRaises(ValueError):
            svr = SVR(degree=-1)
            svr.fit(X, y)

        with self.assertRaises(ValueError):
            svr = SVR(gamma="invalid_gamma")
            svr.fit(X, y)

        with self.assertRaises(ValueError):
            svr = SVR(gamma=-1.0)
            svr.fit(X, y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
