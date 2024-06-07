import unittest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from sklearn.naive_bayes import GaussianNB as SkGNB
from scratchml.models.naive_bayes import GaussianNB
from test.utils import generate_classification_dataset, repeat


class Test_NaiveBayes(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_classification_dataset(
            n_samples=10000, n_classes=2, n_features=5
        )

        gnb = GaussianNB()
        sk_gnb = SkGNB()

        gnb.fit(X, y)
        sk_gnb.fit(X, y)

        pred_gnb = gnb.predict(X)
        pred_sk_gnb = sk_gnb.predict(X)

        pred_proba_gnb = gnb.predict_proba(X)
        pred_proba_sk_gnb = sk_gnb.predict_proba(X)

        pred_log_proba_gnb = gnb.predict_log_proba(X)
        pred_log_proba_sk_gnb = sk_gnb.predict_log_proba(X)

        score = gnb.score(X, y)
        sk_score = sk_gnb.score(X, y)

        assert_equal(gnb.n_features_in_, sk_gnb.n_features_in_)
        assert_equal(type(gnb.class_count_), type(sk_gnb.class_count_))
        assert_equal(gnb.class_prior_, sk_gnb.class_prior_)
        assert_equal(type(gnb.class_prior_), type(sk_gnb.class_prior_))
        assert_almost_equal(gnb.epsilon_, sk_gnb.epsilon_, decimal=4)
        assert_equal(gnb.classes_, sk_gnb.classes_)
        assert_equal(type(gnb.classes_), type(sk_gnb.classes_))
        assert_almost_equal(gnb.var_, sk_gnb.var_, decimal=4)
        assert_equal(gnb.var_.shape, sk_gnb.var_.shape)
        assert_equal(type(gnb.var_), type(sk_gnb.var_))
        assert_almost_equal(gnb.theta_, sk_gnb.theta_, decimal=4)
        assert_equal(gnb.theta_.shape, sk_gnb.theta_.shape)
        assert_equal(type(gnb.theta_), type(sk_gnb.theta_))
        assert_equal(pred_gnb.shape, pred_sk_gnb.shape)
        assert_equal(type(pred_gnb), type(pred_sk_gnb))
        assert np.max(np.abs(pred_gnb - pred_sk_gnb)) < 1
        assert_almost_equal(score, sk_score)
        assert_equal(pred_proba_gnb.shape, pred_proba_sk_gnb.shape)
        assert_equal(type(pred_proba_gnb), type(pred_proba_sk_gnb))
        assert np.max(np.abs(pred_proba_gnb - pred_proba_sk_gnb)) < 1
        assert_equal(pred_log_proba_gnb.shape, pred_log_proba_sk_gnb.shape)
        assert_equal(type(pred_log_proba_gnb), type(pred_log_proba_sk_gnb))
        assert np.max(np.abs(pred_log_proba_gnb - pred_log_proba_sk_gnb)) < 1

    @repeat(10)
    def test_2(self):
        X, y = generate_classification_dataset(
            n_samples=10000, n_classes=8, n_features=10
        )

        gnb = GaussianNB()
        sk_gnb = SkGNB()

        gnb.fit(X, y)
        sk_gnb.fit(X, y)

        pred_gnb = gnb.predict(X)
        pred_sk_gnb = sk_gnb.predict(X)

        pred_proba_gnb = gnb.predict_proba(X)
        pred_proba_sk_gnb = sk_gnb.predict_proba(X)

        pred_log_proba_gnb = gnb.predict_log_proba(X)
        pred_log_proba_sk_gnb = sk_gnb.predict_log_proba(X)

        score = gnb.score(X, y)
        sk_score = sk_gnb.score(X, y)

        assert_equal(gnb.n_features_in_, sk_gnb.n_features_in_)
        assert_equal(type(gnb.class_count_), type(sk_gnb.class_count_))
        assert_equal(gnb.class_prior_, sk_gnb.class_prior_)
        assert_equal(type(gnb.class_prior_), type(sk_gnb.class_prior_))
        assert_almost_equal(gnb.epsilon_, sk_gnb.epsilon_, decimal=4)
        assert_equal(gnb.classes_, sk_gnb.classes_)
        assert_equal(type(gnb.classes_), type(sk_gnb.classes_))
        assert_almost_equal(gnb.var_, sk_gnb.var_, decimal=4)
        assert_equal(gnb.var_.shape, sk_gnb.var_.shape)
        assert_equal(type(gnb.var_), type(sk_gnb.var_))
        assert_almost_equal(gnb.theta_, sk_gnb.theta_, decimal=4)
        assert_equal(gnb.theta_.shape, sk_gnb.theta_.shape)
        assert_equal(type(gnb.theta_), type(sk_gnb.theta_))
        assert_equal(pred_gnb.shape, pred_sk_gnb.shape)
        assert_equal(type(pred_gnb), type(pred_sk_gnb))
        assert np.max(np.abs(pred_gnb - pred_sk_gnb)) < 1
        assert_almost_equal(score, sk_score)
        assert_equal(pred_proba_gnb.shape, pred_proba_sk_gnb.shape)
        assert_equal(type(pred_proba_gnb), type(pred_proba_sk_gnb))
        assert np.max(np.abs(pred_proba_gnb - pred_proba_sk_gnb)) < 1
        assert_equal(pred_log_proba_gnb.shape, pred_log_proba_sk_gnb.shape)
        assert_equal(type(pred_log_proba_gnb), type(pred_log_proba_sk_gnb))
        assert np.max(np.abs(pred_log_proba_gnb - pred_log_proba_sk_gnb)) < 1

    @repeat(10)
    def test_3(self):
        X, y = generate_classification_dataset(
            n_samples=10000, n_classes=2, n_features=5
        )

        gnb = GaussianNB(var_smoothing=1e-05)
        sk_gnb = SkGNB(var_smoothing=1e-05)

        gnb.fit(X, y)
        sk_gnb.fit(X, y)

        pred_gnb = gnb.predict(X)
        pred_sk_gnb = sk_gnb.predict(X)

        pred_proba_gnb = gnb.predict_proba(X)
        pred_proba_sk_gnb = sk_gnb.predict_proba(X)

        pred_log_proba_gnb = gnb.predict_log_proba(X)
        pred_log_proba_sk_gnb = sk_gnb.predict_log_proba(X)

        score = gnb.score(X, y)
        sk_score = sk_gnb.score(X, y)

        assert_equal(gnb.n_features_in_, sk_gnb.n_features_in_)
        assert_equal(type(gnb.class_count_), type(sk_gnb.class_count_))
        assert_equal(gnb.class_prior_, sk_gnb.class_prior_)
        assert_equal(type(gnb.class_prior_), type(sk_gnb.class_prior_))
        assert_almost_equal(gnb.epsilon_, sk_gnb.epsilon_, decimal=4)
        assert_equal(gnb.classes_, sk_gnb.classes_)
        assert_equal(type(gnb.classes_), type(sk_gnb.classes_))
        assert_almost_equal(gnb.var_, sk_gnb.var_, decimal=4)
        assert_equal(gnb.var_.shape, sk_gnb.var_.shape)
        assert_equal(type(gnb.var_), type(sk_gnb.var_))
        assert_almost_equal(gnb.theta_, sk_gnb.theta_, decimal=4)
        assert_equal(gnb.theta_.shape, sk_gnb.theta_.shape)
        assert_equal(type(gnb.theta_), type(sk_gnb.theta_))
        assert_equal(pred_gnb.shape, pred_sk_gnb.shape)
        assert_equal(type(pred_gnb), type(pred_sk_gnb))
        assert np.max(np.abs(pred_gnb - pred_sk_gnb)) < 1
        assert_almost_equal(score, sk_score)
        assert_equal(pred_proba_gnb.shape, pred_proba_sk_gnb.shape)
        assert_equal(type(pred_proba_gnb), type(pred_proba_sk_gnb))
        assert np.max(np.abs(pred_proba_gnb - pred_proba_sk_gnb)) < 1
        assert_equal(pred_log_proba_gnb.shape, pred_log_proba_sk_gnb.shape)
        assert_equal(type(pred_log_proba_gnb), type(pred_log_proba_sk_gnb))
        assert np.max(np.abs(pred_log_proba_gnb - pred_log_proba_sk_gnb)) < 1

    @repeat(10)
    def test_4(self):
        X, y = generate_classification_dataset(
            n_samples=10000, n_classes=5, n_features=5
        )

        _, counts = np.unique(y, return_counts=True)
        counts = np.asarray(counts)
        counts = np.divide(counts, y.shape[0])

        # forcing the sum of the probabilities to be equal than 1
        # workaround to deal with the rounding problem
        if np.sum(counts) != 1:
            diff = np.sum(counts) - 1
            index = np.random.choice(len(counts))
            counts[index] -= diff

        gnb = GaussianNB(priors=counts)
        sk_gnb = SkGNB(priors=counts)

        gnb.fit(X, y)
        sk_gnb.fit(X, y)

        pred_gnb = gnb.predict(X)
        pred_sk_gnb = sk_gnb.predict(X)

        pred_proba_gnb = gnb.predict_proba(X)
        pred_proba_sk_gnb = sk_gnb.predict_proba(X)

        pred_log_proba_gnb = gnb.predict_log_proba(X)
        pred_log_proba_sk_gnb = sk_gnb.predict_log_proba(X)

        score = gnb.score(X, y)
        sk_score = sk_gnb.score(X, y)

        assert_equal(gnb.n_features_in_, sk_gnb.n_features_in_)
        assert_equal(type(gnb.class_count_), type(sk_gnb.class_count_))
        assert_equal(gnb.class_prior_, sk_gnb.class_prior_)
        assert_equal(type(gnb.class_prior_), type(sk_gnb.class_prior_))
        assert_almost_equal(gnb.epsilon_, sk_gnb.epsilon_, decimal=4)
        assert_equal(gnb.classes_, sk_gnb.classes_)
        assert_equal(type(gnb.classes_), type(sk_gnb.classes_))
        assert_almost_equal(gnb.var_, sk_gnb.var_, decimal=4)
        assert_equal(gnb.var_.shape, sk_gnb.var_.shape)
        assert_equal(type(gnb.var_), type(sk_gnb.var_))
        assert_almost_equal(gnb.theta_, sk_gnb.theta_, decimal=4)
        assert_equal(gnb.theta_.shape, sk_gnb.theta_.shape)
        assert_equal(type(gnb.theta_), type(sk_gnb.theta_))
        assert_equal(pred_gnb.shape, pred_sk_gnb.shape)
        assert_equal(type(pred_gnb), type(pred_sk_gnb))
        assert np.max(np.abs(pred_gnb - pred_sk_gnb)) < 1
        assert_almost_equal(score, sk_score)
        assert_equal(pred_proba_gnb.shape, pred_proba_sk_gnb.shape)
        assert_equal(type(pred_proba_gnb), type(pred_proba_sk_gnb))
        assert np.max(np.abs(pred_proba_gnb - pred_proba_sk_gnb)) < 1
        assert_equal(pred_log_proba_gnb.shape, pred_log_proba_sk_gnb.shape)
        assert_equal(type(pred_log_proba_gnb), type(pred_log_proba_sk_gnb))
        assert np.max(np.abs(pred_log_proba_gnb - pred_log_proba_sk_gnb)) < 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
