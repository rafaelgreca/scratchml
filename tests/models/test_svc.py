from sklearn.svm import SVC as SkSVC
from numpy.testing import assert_equal, assert_allclose
from scratchml.models.svc import SVC
from ..utils import repeat, generate_classification_dataset
import unittest
import numpy as np
import math

class Test_SVC(unittest.TestCase):
    """
    Unit test class created to test the custom SVC implementation.
    """

    @repeat(3)
    def test_binary_classification(self):
        """
        Test binary classification and compare custom SVC to Scikit-Learn's SVC.
        """
        X, y = generate_classification_dataset(n_samples=2000, n_features=4, n_classes=2)

        # Fit and predict with scikit-learn SVC
        sk_svc = SkSVC(kernel='linear', max_iter=1000)
        sk_svc.fit(X, y)
        sk_prediction = sk_svc.predict(X)
        sk_score = sk_svc.score(X, y)

        # Fit and predict with custom SVC
        svc = SVC(kernel='linear')
        svc.fit(X, y)
        prediction = svc.predict(X)
        score = svc.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        # Compare predictions and scores
        assert_equal(sk_svc.classes_, svc.classes_)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.01

    @repeat(3)
    def test_multi_class_classification(self):
        """
        Test multi-class classification and compare custom SVC to Scikit-Learn's SVC.
        """
        X, y = generate_classification_dataset(n_samples=2000, n_features=4, n_classes=3)

        # Fit and predict with scikit-learn SVC
        sk_svc = SkSVC(kernel='linear', max_iter=1000)
        sk_svc.fit(X, y)
        sk_prediction = sk_svc.predict(X)
        sk_score = sk_svc.score(X, y)

        # Fit and predict with custom SVC
        svc = SVC(kernel='linear')
        svc.fit(X, y)
        prediction = svc.predict(X)
        score = svc.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        # Compare predictions and scores
        assert_equal(sk_svc.classes_, svc.classes_)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.01

    @repeat(3)
    def test_rbf_kernel(self):
        """
        Test the SVC implementation with RBF kernel and compare it to Scikit-Learn's SVC.
        """
        X, y = generate_classification_dataset(n_samples=2000, n_features=4, n_classes=2)

        # Fit and predict with scikit-learn SVC
        sk_svc = SkSVC(kernel='rbf', max_iter=1000)
        sk_svc.fit(X, y)
        sk_prediction = sk_svc.predict(X)
        sk_score = sk_svc.score(X, y)

        # Fit and predict with custom SVC
        svc = SVC(kernel='rbf')
        svc.fit(X, y)
        prediction = svc.predict(X)
        score = svc.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        # Compare predictions and scores
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.01

    def test_custom_kernel_initialization(self):
        """
        Test that the SVC model can initialize with a custom kernel.
        """
        svc = SVC(kernel='poly')
        self.assertEqual(svc.kernel, 'poly', "Model should be initialized with 'poly' kernel.")

    def test_untrained_model_prediction_error(self):
        """
        Test that predicting with an untrained model raises an error.
        """
        svc = SVC(kernel='linear')
        X, _ = generate_classification_dataset(n_samples=10, n_features=2, n_classes=2)
        
        with self.assertRaises(ValueError):
            svc.predict(X)

if __name__ == '__main__':
    unittest.main()
