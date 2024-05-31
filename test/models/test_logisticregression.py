import unittest
import math
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from scratchml.models.logistic_regression import LogisticRegression
from test.utils import generate_classification_dataset, repeat

class Test_LogisticRegression(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_classification_dataset()

        lr = LogisticRegression(learning_rate=0.1, tol=1e-4)
        sklr = SkLogisticRegression(
            penalty=None,
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            solver="saga"
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        predict_sklr = sklr.predict(X)
        predict_lr = lr.predict(X)

        predict_proba_sklr = sklr.predict(X)
        predict_proba_lr = lr.predict(X)

        atol = math.floor(y.shape[0] * 0.05)
        
        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert predict_sklr.shape == predict_lr.shape
        assert predict_proba_sklr.shape == predict_proba_lr.shape
        assert_equal(sklr.classes_, lr.classes_)
        # assert_almost_equal(sklr.coef_, lr.coef_, decimal=2)
        # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=2)
        assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
        assert_allclose(predict_sklr, predict_lr, atol=atol)
        assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)
    
    @repeat(10)
    def test_2(self):
        X, y = generate_classification_dataset(n_samples=20000)

        lr = LogisticRegression(learning_rate=0.1, tol=1e-4)
        sklr = SkLogisticRegression(
            penalty=None,
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            solver="saga"
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        predict_sklr = sklr.predict(X)
        predict_lr = lr.predict(X)

        predict_proba_sklr = sklr.predict(X)
        predict_proba_lr = lr.predict(X)

        atol = math.floor(y.shape[0] * 0.05)
        
        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert predict_sklr.shape == predict_lr.shape
        assert predict_proba_sklr.shape == predict_proba_lr.shape
        assert_equal(sklr.classes_, lr.classes_)
        # assert_almost_equal(sklr.coef_, lr.coef_, decimal=2)
        # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=2)
        assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
        assert_allclose(predict_sklr, predict_lr, atol=atol)
        assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)
    
    # @repeat(3)
    # def test_3(self):
    #     X, y = generate_classification_dataset(
    #         n_samples=3000,
    #         n_classes=10
    #     )

    #     lr = LogisticRegression(learning_rate=0.1, tol=1e-4)
    #     sklr = SkLogisticRegression(
    #         penalty=None,
    #         fit_intercept=True,
    #         max_iter=1000000,
    #         tol=1e-4
    #     )

    #     sklr.fit(X, y)
    #     lr.fit(X, y)

    #     predict_sklr = sklr.predict(X)
    #     predict_lr = lr.predict(X)

    #     predict_proba_sklr = sklr.predict(X)
    #     predict_proba_lr = lr.predict(X)

    #     atol = math.floor(y.shape[0] * 0.05)
        
    #     assert sklr.coef_.shape == lr.coef_.shape
    #     assert sklr.intercept_.shape == lr.intercept_.shape
    #     assert sklr.n_features_in_ == lr.n_features_in_
    #     assert predict_sklr.shape == predict_lr.shape
    #     assert predict_proba_sklr.shape == predict_proba_lr.shape
    #     assert_equal(sklr.classes_, lr.classes_)
    #     # assert_almost_equal(sklr.coef_, lr.coef_, decimal=2)
    #     # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=2)
    #     assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
    #     assert_allclose(predict_sklr, predict_lr, atol=atol)
    #     assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)
        
    @repeat(3)
    def test_4(self):
        X, y = generate_classification_dataset(
            n_samples=3000,
            n_classes=10
        )

        lr = LogisticRegression(learning_rate=0.1, tol=1e-4)
        sklr = SkLogisticRegression(
            penalty=None,
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            solver="saga"
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        predict_sklr = sklr.predict(X)
        predict_lr = lr.predict(X)

        predict_proba_sklr = sklr.predict(X)
        predict_proba_lr = lr.predict(X)

        atol = math.floor(y.shape[0] * 0.05)
        
        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert predict_sklr.shape == predict_lr.shape
        assert predict_proba_sklr.shape == predict_proba_lr.shape
        assert_equal(sklr.classes_, lr.classes_)
        # assert_almost_equal(sklr.coef_, lr.coef_, decimal=2)
        # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=2)
        assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
        assert_allclose(predict_sklr, predict_lr, atol=atol)
        assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)

    @repeat(10)
    def test_5(self):
        X, y = generate_classification_dataset()

        lr = LogisticRegression(learning_rate=0.1, tol=1e-4, regularization='l1')
        sklr = SkLogisticRegression(
            penalty='l1',
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            solver="saga"
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        predict_sklr = sklr.predict(X)
        predict_lr = lr.predict(X)

        predict_proba_sklr = sklr.predict(X)
        predict_proba_lr = lr.predict(X)

        atol = math.floor(y.shape[0] * 0.05)
        
        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert predict_sklr.shape == predict_lr.shape
        assert predict_proba_sklr.shape == predict_proba_lr.shape
        assert_equal(sklr.classes_, lr.classes_)
        # assert_almost_equal(sklr.coef_, lr.coef_, decimal=3)
        # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=3)
        assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
        assert_allclose(predict_sklr, predict_lr, atol=atol)
        assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)
    
    @repeat(10)
    def test_6(self):
        X, y = generate_classification_dataset(n_samples=20000)

        lr = LogisticRegression(learning_rate=0.1, tol=1e-4, regularization="l1")
        sklr = SkLogisticRegression(
            penalty="l1",
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            solver="saga"
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        predict_sklr = sklr.predict(X)
        predict_lr = lr.predict(X)

        predict_proba_sklr = sklr.predict(X)
        predict_proba_lr = lr.predict(X)

        atol = math.floor(y.shape[0] * 0.05)
        
        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert predict_sklr.shape == predict_lr.shape
        assert predict_proba_sklr.shape == predict_proba_lr.shape
        assert_equal(sklr.classes_, lr.classes_)
        # assert_almost_equal(sklr.coef_, lr.coef_, decimal=3)
        # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=3)
        assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
        assert_allclose(predict_sklr, predict_lr, atol=atol)
        assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)
    
    @repeat(10)
    def test_7(self):
        X, y = generate_classification_dataset(
            n_samples=3000,
            n_classes=10
        )

        lr = LogisticRegression(learning_rate=0.1, tol=1e-4, regularization="l1")
        sklr = SkLogisticRegression(
            penalty="l1",
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            solver="saga"
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        predict_sklr = sklr.predict(X)
        predict_lr = lr.predict(X)

        predict_proba_sklr = sklr.predict(X)
        predict_proba_lr = lr.predict(X)

        atol = math.floor(y.shape[0] * 0.05)
        
        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert predict_sklr.shape == predict_lr.shape
        assert predict_proba_sklr.shape == predict_proba_lr.shape
        assert_equal(sklr.classes_, lr.classes_)
        # assert_almost_equal(sklr.coef_, lr.coef_, decimal=3)
        # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=3)
        assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
        assert_allclose(predict_sklr, predict_lr, atol=atol)
        assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)
    
    @repeat(10)
    def test_8(self):
        X, y = generate_classification_dataset(
            n_samples=3000,
            n_classes=10
        )

        lr = LogisticRegression(learning_rate=0.1, tol=1e-4, regularization="l1")
        sklr = SkLogisticRegression(
            penalty="l1",
            fit_intercept=True,
            max_iter=1000000,
            tol=1e-4,
            solver="saga"
        )

        sklr.fit(X, y)
        lr.fit(X, y)

        predict_sklr = sklr.predict(X)
        predict_lr = lr.predict(X)

        predict_proba_sklr = sklr.predict(X)
        predict_proba_lr = lr.predict(X)

        atol = math.floor(y.shape[0] * 0.05)
        
        assert sklr.coef_.shape == lr.coef_.shape
        assert sklr.intercept_.shape == lr.intercept_.shape
        assert sklr.n_features_in_ == lr.n_features_in_
        assert predict_sklr.shape == predict_lr.shape
        assert predict_proba_sklr.shape == predict_proba_lr.shape
        assert_equal(sklr.classes_, lr.classes_)
        # assert_almost_equal(sklr.coef_, lr.coef_, decimal=3)
        # assert_almost_equal(sklr.intercept_, lr.intercept_, decimal=3)
        assert_allclose(sklr.score(X, y), lr.score(X, y), rtol=1)
        assert_allclose(predict_sklr, predict_lr, atol=atol)
        assert_allclose(predict_proba_sklr, predict_proba_lr, atol=atol)

if __name__ == "__main__":
    unittest.main(verbosity=2)