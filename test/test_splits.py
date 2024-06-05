import unittest
import random
import numpy as np
from sklearn.model_selection import train_test_split as SkTTS
from sklearn.model_selection import KFold as SkKF
from sklearn.model_selection import StratifiedKFold as SkSKF
from scratchml.utils import train_test_split, KFold
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_array_equal,
)
from test.utils import repeat, generate_classification_dataset


class Test_DataSplits(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        test_size = random.uniform(0, 1)

        X, y = generate_classification_dataset(n_samples=50000, n_classes=10)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, stratify=False
        )

        SkX_train, SkX_test, Sky_train, Sky_test = SkTTS(
            X, y, test_size=test_size, shuffle=False, stratify=None
        )

        assert_equal(type(X_train), type(SkX_train))
        assert_equal(type(X_test), type(SkX_test))
        assert_equal(type(y_train), type(Sky_train))
        assert_equal(type(y_test), type(Sky_test))
        assert_almost_equal(test_size, X_test.shape[0] / X.shape[0], decimal=3)
        assert_equal(X_train.shape[0] + X_test.shape[0], X.shape[0])
        assert_equal(y_train.shape[0] + y_test.shape[0], y.shape[0])
        assert np.abs(X_train.shape[0] - SkX_train.shape[0]) < X.shape[0] * 0.005
        assert np.abs(X_test.shape[0] - SkX_test.shape[0]) < X.shape[0] * 0.005
        assert np.abs(y_train.shape[0] - Sky_train.shape[0]) < y.shape[0] * 0.005
        assert np.abs(y_test.shape[0] - Sky_test.shape[0]) < y.shape[0] * 0.005

        SX_train, SX_test, Sy_train, Sy_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=False
        )

        SSkX_train, SSkX_test, SSky_train, SSky_test = SkTTS(
            X, y, test_size=test_size, shuffle=False, stratify=None
        )

        assert_equal(type(SX_train), type(SSkX_train))
        assert_equal(type(SX_test), type(SSkX_test))
        assert_equal(type(Sy_train), type(SSky_train))
        assert_equal(type(Sy_test), type(SSky_test))
        assert_raises(AssertionError, assert_array_equal, SX_train, X_train)
        assert_raises(AssertionError, assert_array_equal, SX_test, X_test)
        assert_raises(AssertionError, assert_array_equal, Sy_train, y_train)
        assert_raises(AssertionError, assert_array_equal, Sy_test, y_test)
        assert_almost_equal(test_size, SX_test.shape[0] / X.shape[0], decimal=3)
        assert_equal(SX_train.shape[0] + SX_test.shape[0], X.shape[0])
        assert_equal(Sy_train.shape[0] + Sy_test.shape[0], y.shape[0])
        assert np.abs(SX_train.shape[0] - SSkX_train.shape[0]) < X.shape[0] * 0.005
        assert np.abs(SX_test.shape[0] - SSkX_test.shape[0]) < X.shape[0] * 0.005
        assert np.abs(Sy_train.shape[0] - SSky_train.shape[0]) < y.shape[0] * 0.005
        assert np.abs(Sy_test.shape[0] - SSky_test.shape[0]) < y.shape[0] * 0.005

        SSX_train, SSX_test, SSy_train, SSy_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=True
        )

        SSSkX_train, SSSkX_test, SSSky_train, SSSky_test = SkTTS(
            X, y, test_size=test_size, shuffle=True, stratify=y
        )

        # calculating the classes distribution for the entire dataset
        unique, counts = np.unique(y, return_counts=True)
        counts = np.asarray((unique, counts)).T
        sample_classes_distribution = [(u, c / y.shape[0]) for u, c in counts]

        # calculating the classes distribution for the stratified y train set
        unique, counts = np.unique(SSy_train, return_counts=True)
        counts = np.asarray((unique, counts)).T
        train_classes_distribution = [(u, c / SSy_train.shape[0]) for u, c in counts]

        # calculating the classes distribution for the stratified y test set
        unique, counts = np.unique(SSy_test, return_counts=True)
        counts = np.asarray((unique, counts)).T
        test_classes_distribution = [(u, c / SSy_test.shape[0]) for u, c in counts]

        # calculating the differences between the samples and train distribution
        train_ddiff = [
            np.abs(s[1] - t[1])
            for s, t in zip(sample_classes_distribution, train_classes_distribution)
        ]

        # calculating the differences between the samples and test distribution
        test_ddiff = [
            np.abs(s[1] - t[1])
            for s, t in zip(sample_classes_distribution, test_classes_distribution)
        ]

        assert_equal(type(SSX_train), type(SSSkX_train))
        assert_equal(type(SSX_test), type(SSSkX_test))
        assert_equal(type(SSy_train), type(SSSky_train))
        assert_equal(type(SSy_test), type(SSSky_test))
        assert_almost_equal(test_size, SSX_test.shape[0] / X.shape[0], decimal=3)
        assert_equal(SSX_train.shape[0] + SSX_test.shape[0], X.shape[0])
        assert_equal(SSy_train.shape[0] + SSy_test.shape[0], y.shape[0])
        assert np.abs(SSX_train.shape[0] - SSSkX_train.shape[0]) < X.shape[0] * 0.005
        assert np.abs(SSX_test.shape[0] - SSSkX_test.shape[0]) < X.shape[0] * 0.005
        assert np.abs(SSy_train.shape[0] - SSSky_train.shape[0]) < y.shape[0] * 0.005
        assert np.abs(SSy_test.shape[0] - SSSky_test.shape[0]) < y.shape[0] * 0.005
        assert np.max(train_ddiff) < 0.01
        assert np.max(test_ddiff) < 0.01

    @repeat(10)
    def test_2(self):
        X, y = generate_classification_dataset(n_samples=50000, n_classes=10)

        folds = KFold(X=X, y=y, stratify=False, shuffle=True, n_splits=3)

        skf = SkKF(n_splits=3, shuffle=True)
        skfolds = [
            [train_index, test_index] for (train_index, test_index) in skf.split(X)
        ]

        assert_equal(len(folds), len(skfolds))

        for fold, skfold in zip(folds, skfolds):
            train_index, test_index = fold
            sktrain_index, sktest_index = skfold

            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            Sk_X_train = X[sktrain_index]
            Sk_y_train = X[sktrain_index]
            Sk_X_test = X[sktest_index]
            Sk_y_test = X[sktest_index]

            assert_equal(type(train_index), type(sktrain_index))
            assert_equal(type(test_index), type(sktest_index))
            assert_equal(X_train.shape[0] + X_test.shape[0], X.shape[0])
            assert_equal(y_train.shape[0] + y_test.shape[0], y.shape[0])
            assert np.abs(X_train.shape[0] - Sk_X_train.shape[0]) < X.shape[0] * 0.005
            assert np.abs(X_test.shape[0] - Sk_X_test.shape[0]) < X.shape[0] * 0.005
            assert np.abs(y_train.shape[0] - Sk_y_train.shape[0]) < y.shape[0] * 0.005
            assert np.abs(y_test.shape[0] - Sk_y_test.shape[0]) < y.shape[0] * 0.005

        folds = KFold(X=X, y=y, stratify=True, shuffle=False, n_splits=3)

        sskf = SkSKF(n_splits=3, shuffle=True)
        skfolds = [
            [train_index, test_index] for (train_index, test_index) in sskf.split(X, y)
        ]

        assert_equal(len(folds), len(skfolds))

        # calculating the classes distribution for the entire dataset
        unique, counts = np.unique(y, return_counts=True)
        counts = np.asarray((unique, counts)).T
        sample_classes_distribution = [(u, c / y.shape[0]) for u, c in counts]

        for fold, skfold in zip(folds, skfolds):
            train_index, test_index = fold
            sktrain_index, sktest_index = skfold

            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            Sk_X_train = X[sktrain_index]
            Sk_y_train = X[sktrain_index]
            Sk_X_test = X[sktest_index]
            Sk_y_test = X[sktest_index]

            assert_equal(type(train_index), type(sktrain_index))
            assert_equal(type(test_index), type(sktest_index))
            assert_equal(X_train.shape[0] + X_test.shape[0], X.shape[0])
            assert_equal(y_train.shape[0] + y_test.shape[0], y.shape[0])
            assert np.abs(X_train.shape[0] - Sk_X_train.shape[0]) < X.shape[0] * 0.005
            assert np.abs(X_test.shape[0] - Sk_X_test.shape[0]) < X.shape[0] * 0.005
            assert np.abs(y_train.shape[0] - Sk_y_train.shape[0]) < y.shape[0] * 0.005
            assert np.abs(y_test.shape[0] - Sk_y_test.shape[0]) < y.shape[0] * 0.005

            # calculating the classes distribution for the stratified y train set
            unique, counts = np.unique(y_train, return_counts=True)
            counts = np.asarray((unique, counts)).T
            train_classes_distribution = [(u, c / y_train.shape[0]) for u, c in counts]

            # calculating the classes distribution for the stratified y test set
            unique, counts = np.unique(y_test, return_counts=True)
            counts = np.asarray((unique, counts)).T
            test_classes_distribution = [(u, c / y_test.shape[0]) for u, c in counts]

            # calculating the differences between the samples and train distribution
            train_ddiff = [
                np.abs(s[1] - t[1])
                for s, t in zip(sample_classes_distribution, train_classes_distribution)
            ]

            # calculating the differences between the samples and test distribution
            test_ddiff = [
                np.abs(s[1] - t[1])
                for s, t in zip(sample_classes_distribution, test_classes_distribution)
            ]

            assert np.max(train_ddiff) < 0.01
            assert np.max(test_ddiff) < 0.01


if __name__ == "__main__":
    unittest.main(verbosity=2)
