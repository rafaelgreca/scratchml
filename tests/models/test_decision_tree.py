from sklearn.tree import DecisionTreeClassifier as SkDTC
from sklearn.tree import DecisionTreeRegressor as SkDTR
from numpy.testing import assert_equal, assert_allclose
from scratchml.models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import repeat, generate_classification_dataset, generate_regression_dataset
import unittest
import math


class Test_Decision_Tree(unittest.TestCase):
    """
    Unittest class created to test the Decision Tree implementation.
    """

    @repeat(5)
    def test_1(self):
        """
        Test the Decision Tree Classifier implementation on a small dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=2
        )

        sk_dt = SkDTC(max_depth=100)
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100)
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_2(self):
        """
        Test the Decision Tree Classifier implementation on a bigger dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=8000, n_features=6, n_classes=2
        )

        sk_dt = SkDTC(max_depth=100)
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100)
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_3(self):
        """
        Test the Decision Tree Classifier implementation on a small multi-class dataset
        using all features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_dt = SkDTC(max_depth=100)
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100)
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_4(self):
        """
        Test the Decision Tree Classifier implementation on a small dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=2
        )

        sk_dt = SkDTC(max_depth=100, max_features="sqrt")
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100, max_features="sqrt")
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_5(self):
        """
        Test the Decision Tree Classifier implementation on a bigger dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=8000, n_features=6, n_classes=2
        )

        sk_dt = SkDTC(max_depth=100, max_features="sqrt")
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100, max_features="sqrt")
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_6(self):
        """
        Test the Decision Tree Classifier implementation on a small multi-class dataset
        using 'sqrt' features and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_dt = SkDTC(max_depth=100, max_features="sqrt")
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100, max_features="sqrt")
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_7(self):
        """
        Test the Decision Tree Classifier implementation on a small dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=2
        )

        sk_dt = SkDTC(max_depth=100, min_samples_split=5)
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100, min_samples_split=5)
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_8(self):
        """
        Test the Decision Tree Classifier implementation on a bigger dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=8000, n_features=6, n_classes=2
        )

        sk_dt = SkDTC(max_depth=100, min_samples_split=5)
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100, min_samples_split=5)
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    @repeat(5)
    def test_9(self):
        """
        Test the Decision Tree Classifier implementation on a small multi-class dataset
        with higher min_samples_split and then compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=3, n_classes=5
        )

        sk_dt = SkDTC(max_depth=100, min_samples_split=5)
        sk_dt.fit(X, y)
        sk_prediction = sk_dt.predict(X)

        dt = DecisionTreeClassifier(max_depth=100, min_samples_split=5)
        dt.fit(X, y)
        prediction = dt.predict(X)

        atol = math.floor(y.shape[0] * 0.1)

        assert_equal(sk_dt.classes_, dt.classes_)
        assert_equal(sk_dt.n_classes_, dt.n_classes_)
        assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
        assert_equal(sk_dt.max_features_, dt.max_features_)
        assert_equal(type(sk_prediction), type(prediction))
        assert_equal(sk_prediction.shape, prediction.shape)
        assert_allclose(sk_prediction, prediction, atol=atol)

    # @repeat(5)
    # def test_10(self):
    #     """
    #     Test the Decision Tree Regressor implementation on a small dataset
    #     using all features and then compares it to the Scikit-Learn implementation.
    #     """
    #     X, y = generate_regression_dataset(n_samples=2000, n_features=3)

    #     sk_dt = SkDTR(max_depth=100)
    #     sk_dt.fit(X, y)
    #     sk_prediction = sk_dt.predict(X)

    #     dt = DecisionTreeRegressor(max_depth=100)
    #     dt.fit(X, y)
    #     prediction = dt.predict(X)

    #     atol = math.floor(y.shape[0] * 0.1)

    #     assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
    #     assert_equal(sk_dt.max_features_, dt.max_features_)
    #     assert_equal(type(sk_prediction), type(prediction))
    #     assert_equal(sk_prediction.shape, prediction.shape)
    #     assert_allclose(sk_prediction, prediction, atol=atol)

    # @repeat(5)
    # def test_11(self):
    #     """
    #     Test the Decision Tree Regressor implementation on a bigger dataset
    #     using all features and then compares it to the Scikit-Learn implementation.
    #     """
    #     X, y = generate_regression_dataset(n_samples=8000, n_features=6)

    #     sk_dt = SkDTR(max_depth=100)
    #     sk_dt.fit(X, y)
    #     sk_prediction = sk_dt.predict(X)

    #     dt = DecisionTreeRegressor(max_depth=100)
    #     dt.fit(X, y)
    #     prediction = dt.predict(X)

    #     atol = math.floor(y.shape[0] * 0.1)

    #     assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
    #     assert_equal(sk_dt.max_features_, dt.max_features_)
    #     assert_equal(type(sk_prediction), type(prediction))
    #     assert_equal(sk_prediction.shape, prediction.shape)
    #     assert_allclose(sk_prediction, prediction, atol=atol)

    # @repeat(5)
    # def test_12(self):
    #     """
    #     Test the Decision Tree Regressor implementation on a small dataset
    #     using 'sqrt' features and then compares it to the Scikit-Learn implementation.
    #     """
    #     X, y = generate_regression_dataset(n_samples=2000, n_features=3)

    #     sk_dt = SkDTR(max_depth=100, max_features="sqrt")
    #     sk_dt.fit(X, y)
    #     sk_prediction = sk_dt.predict(X)

    #     dt = DecisionTreeRegressor(max_depth=100, max_features="sqrt")
    #     dt.fit(X, y)
    #     prediction = dt.predict(X)

    #     atol = math.floor(y.shape[0] * 0.1)

    #     assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
    #     assert_equal(sk_dt.max_features_, dt.max_features_)
    #     assert_equal(type(sk_prediction), type(prediction))
    #     assert_equal(sk_prediction.shape, prediction.shape)
    #     assert_allclose(sk_prediction, prediction, atol=atol)

    # @repeat(5)
    # def test_13(self):
    #     """
    #     Test the Decision Tree Regressor implementation on a bigger dataset
    #     using 'sqrt' features and then compares it to the Scikit-Learn implementation.
    #     """
    #     X, y = generate_regression_dataset(n_samples=8000, n_features=6)

    #     sk_dt = SkDTR(max_depth=100, max_features="sqrt")
    #     sk_dt.fit(X, y)
    #     sk_prediction = sk_dt.predict(X)

    #     dt = DecisionTreeRegressor(max_depth=100, max_features="sqrt")
    #     dt.fit(X, y)
    #     prediction = dt.predict(X)

    #     atol = math.floor(y.shape[0] * 0.1)

    #     assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
    #     assert_equal(sk_dt.max_features_, dt.max_features_)
    #     assert_equal(type(sk_prediction), type(prediction))
    #     assert_equal(sk_prediction.shape, prediction.shape)
    #     assert_allclose(sk_prediction, prediction, atol=atol)

    # @repeat(5)
    # def test_14(self):
    #     """
    #     Test the Decision Tree Regressor implementation on a small dataset
    #     with higher min_samples_split and then compares it to the Scikit-Learn implementation.
    #     """
    #     X, y = generate_regression_dataset(n_samples=2000, n_features=3)

    #     sk_dt = SkDTR(max_depth=100, min_samples_split=5)
    #     sk_dt.fit(X, y)
    #     sk_prediction = sk_dt.predict(X)

    #     dt = DecisionTreeRegressor(max_depth=100, min_samples_split=5)
    #     dt.fit(X, y)
    #     prediction = dt.predict(X)

    #     atol = math.floor(y.shape[0] * 0.1)

    #     assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
    #     assert_equal(sk_dt.max_features_, dt.max_features_)
    #     assert_equal(type(sk_prediction), type(prediction))
    #     assert_equal(sk_prediction.shape, prediction.shape)
    #     assert_allclose(sk_prediction, prediction, atol=atol)

    # @repeat(5)
    # def test_15(self):
    #     """
    #     Test the Decision Tree Regressor implementation on a bigger dataset
    #     with higher min_samples_split and then compares it to the Scikit-Learn implementation.
    #     """
    #     X, y = generate_regression_dataset(n_samples=8000, n_features=6)

    #     sk_dt = SkDTR(max_depth=100, min_samples_split=5)
    #     sk_dt.fit(X, y)
    #     sk_prediction = sk_dt.predict(X)

    #     dt = DecisionTreeRegressor(max_depth=100, min_samples_split=5)
    #     dt.fit(X, y)
    #     prediction = dt.predict(X)

    #     atol = math.floor(y.shape[0] * 0.1)

    #     assert_equal(sk_dt.n_features_in_, dt.n_features_in_)
    #     assert_equal(sk_dt.max_features_, dt.max_features_)
    #     assert_equal(type(sk_prediction), type(prediction))
    #     assert_equal(sk_prediction.shape, prediction.shape)
    #     assert_allclose(sk_prediction, prediction, atol=atol)
