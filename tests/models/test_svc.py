from numpy.testing import assert_allclose, assert_equal
from sklearn.svm import SVC as SkSVC
from scratchml.models.svc import SVC
from scratchml.scalers import StandardScaler
from ..utils import generate_classification_dataset, repeat
import unittest
import numpy as np
import math


class Test_SVC(unittest.TestCase):
    """
    Unit test class for the custom SVC implementation.
    """

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
        assert abs(sklearn_score - custom_score) / abs(sklearn_score) < 0.05

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
        assert abs(sklearn_score - custom_score) / abs(sklearn_score) < 0.05

    @repeat(3)
    def test_rbf_kernel(self):
        """
        Test the custom SVC with RBF kernel against Scikit-Learn's SVC.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )

        custom_svc = SVC(kernel="rbf")
        sklearn_svc = SkSVC(kernel="rbf", max_iter=1000)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        custom_score = custom_svc.score(X, y)
        sklearn_score = sklearn_svc.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)
        assert_allclose(sklearn_pred, custom_pred, atol=atol)
        assert abs(sklearn_score - custom_score) / abs(sklearn_score) < 0.05

    def test_untrained_model_prediction_error(self):
        """
        Ensure an error is raised when predicting with an untrained model.
        """
        svc = SVC(kernel="linear")
        X, _ = generate_classification_dataset(n_samples=10, n_features=2, n_classes=2)

        with self.assertRaises(ValueError):
            svc.predict(X)

    def test_custom_kernel_initialization(self):
        """
        Ensure the SVC model initializes correctly with a custom kernel.
        """
        svc = SVC(kernel="poly")
        self.assertEqual(
            svc.kernel, "poly", "Model should initialize with 'poly' kernel."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
