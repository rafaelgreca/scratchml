from sklearn.ensemble import RandomForestClassifier as SkRFC
from sklearn.ensemble import RandomForestRegressor as SkRFR
from numpy.testing import assert_equal, assert_allclose
from scratchml.models.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from ..utils import repeat, generate_classification_dataset, generate_regression_dataset
import unittest
import math
import numpy as np


class Test_Random_Forest(unittest.TestCase):
    """
    Unittest class created to test the Random Forest implementations.
    """

    @repeat(5)
    def test_1(self):
        """
        Test the Random Forest Classifier implementation on a small dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=2
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_2(self):
        """
        Test the Random Forest Classifier implementation on a bigger dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=8000, n_features=6, n_classes=2
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_3(self):
        """
        Test the Random Forest Classifier implementation on a small multi-class dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_4(self):
        """
        Test the Random Forest Classifier implementation on a small dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=2
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10, max_features="sqrt")
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10, max_features="sqrt")
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_5(self):
        """
        Test the Random Forest Classifier implementation on a bigger dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=8000, n_features=6, n_classes=2
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10, max_features="sqrt")
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10, max_features="sqrt")
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_6(self):
        """
        Test the Random Forest Classifier implementation on a small multi-class dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10, max_features="sqrt")
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10, max_features="sqrt")
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_7(self):
        """
        Test the Random Forest Classifier implementation on a small dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=2
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10, min_samples_split=5)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_8(self):
        """
        Test the Random Forest Classifier implementation on a bigger dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=8000, n_features=6, n_classes=2
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10, min_samples_split=5)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_9(self):
        """
        Test the Random Forest Classifier implementation on a small multi-class dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_rf = SkRFC(n_estimators=10, max_depth=10, min_samples_split=5)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)
        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_10(self):
        """
        Test the Random Forest Classifier implementation on a small multi-class dataset,
        higher min_samples_split, entropy criterion and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_rf = SkRFC(
            n_estimators=10, max_depth=10, min_samples_split=5, criterion="entropy"
        )
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(
            n_estimators=10, max_depth=10, min_samples_split=5, criterion="entropy"
        )
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)
        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_11(self):
        """
        Test the Random Forest Classifier implementation on a small multi-class dataset,
        higher min_samples_split, log_loss criterion and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_rf = SkRFC(
            n_estimators=10, max_depth=10, min_samples_split=5, criterion="log_loss"
        )
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(
            n_estimators=10, max_depth=10, min_samples_split=5, criterion="log_loss"
        )
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)
        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_12(self):
        """
        Test the Random Forest Classifier implementation on a small multi-class dataset,
        higher min_samples_split, higher n_estimators and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_rf = SkRFC(n_estimators=100, max_depth=10, min_samples_split=5)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)
        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.classes_, rf.classes_)
        assert_equal(sk_rf.n_classes_, rf.n_classes_)
        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_13(self):
        """
        Test the Random Forest Regressor implementation on a small dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=2000, n_features=3)

        sk_rf = SkRFR(n_estimators=10, max_depth=10)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(n_estimators=10, max_depth=10)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_14(self):
        """
        Test the Random Forest Regressor implementation on a bigger dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=8000, n_features=6)

        sk_rf = SkRFR(n_estimators=10, max_depth=10)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(n_estimators=10, max_depth=10)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_15(self):
        """
        Test the Random Forest Regressor implementation on a small dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=2000, n_features=3)

        sk_rf = SkRFR(n_estimators=10, max_depth=10, max_features="sqrt")
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(n_estimators=10, max_depth=10, max_features="sqrt")
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_16(self):
        """
        Test the Random Forest Regressor implementation on a bigger dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=8000, n_features=6)

        sk_rf = SkRFR(n_estimators=10, max_depth=10, max_features="sqrt")
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(n_estimators=10, max_depth=10, max_features="sqrt")
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_17(self):
        """
        Test the Random Forest Regressor implementation on a small dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=2000, n_features=3)

        sk_rf = SkRFR(n_estimators=10, max_depth=10, min_samples_split=5)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_18(self):
        """
        Test the Random Forest Regressor implementation on a bigger dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=8000, n_features=6)

        sk_rf = SkRFR(n_estimators=10, max_depth=10, min_samples_split=5)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_19(self):
        """
        Test the Random Forest Regressor implementation on a bigger dataset, higher
        min_samples_split, poisson criteria and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=8000, n_features=6)
        y = np.abs(y)  # Poisson does not allow negative values

        sk_rf = SkRFR(
            n_estimators=10, max_depth=10, min_samples_split=5, criterion="poisson"
        )
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(
            n_estimators=10, max_depth=10, min_samples_split=5, criterion="poisson"
        )
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_20(self):
        """
        Test the Random Forest Regressor implementation on a bigger dataset, higher
        min_samples_split, mean absolute error criteria and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=8000, n_features=6)

        sk_rf = SkRFR(
            n_estimators=10,
            max_depth=10,
            min_samples_split=5,
            criterion="absolute_error",
        )
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=10,
            min_samples_split=5,
            criterion="absolute_error",
        )
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05

    @repeat(5)
    def test_21(self):
        """
        Test the Random Forest Regressor implementation on a bigger dataset, higher
        min_samples_split, higher n_estimators and then compares it
        to the Scikit-Learn implementation.
        """
        X, y = generate_regression_dataset(n_samples=8000, n_features=6)

        sk_rf = SkRFR(n_estimators=100, max_depth=10, min_samples_split=5)
        sk_rf.fit(X, y)
        sk_prediction = sk_rf.predict(X)
        sk_score = sk_rf.score(X, y)

        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        prediction = rf.predict(X)
        score = rf.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_rf.n_features_in_, rf.n_features_in_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)
        assert np.abs(sk_score - score) / np.abs(sk_score) < 0.05
