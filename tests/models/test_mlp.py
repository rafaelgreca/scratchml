from numpy.testing import assert_allclose, assert_equal
from sklearn.neural_network import MLPClassifier as SkMLP
from scratchml.models.multilayer_perceptron import MLPClassifier
from ..utils import generate_classification_dataset, repeat
import unittest
import math
import numpy as np


class Test_MLP(unittest.TestCase):
    """
    Unittest class created to test the MLP implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the Multilayer Perceptron implementation and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(n_features=3, n_classes=2)

        mlp = MLPClassifier(loss_function="bce", hidden_layer_sizes=(32, 64))
        skmlp = SkMLP(
            hidden_layer_sizes=(32, 64),
            solver="sgd",
            early_stopping=False,
            n_iter_no_change=200,
        )

        skmlp.fit(X, y)
        mlp.fit(X, y)

        predict_skmlp = skmlp.predict(X)
        predict_mlp = mlp.predict(X).squeeze()

        predict_proba_skmlp = skmlp.predict_proba(X)
        predict_proba_mlp = mlp.predict_proba(X).squeeze()

        print(predict_proba_mlp.shape, predict_proba_skmlp.shape)

        score_skmlp = skmlp.score(X, y)
        score = mlp.score(X, y)

        atol = math.floor(y.shape[0] * 0.02)

        assert mlp.n_features_in_ == skmlp.n_features_in_
        assert mlp.n_layers_ == skmlp.n_layers_
        assert mlp.n_outputs_ == skmlp.n_outputs_

        if mlp.out_activation_ == "sigmoid":
            # sklearn refers the "sigmoid" activation function as "logistic"
            assert "logistic" == skmlp.out_activation_

        assert len(mlp.coefs_) == len(skmlp.coefs_)

        for mc, skmc in zip(mlp.coefs_, skmlp.coefs_):
            assert mc.shape == skmc.shape
            # assert_allclose(mc, skmc)

        for mi, skmi in zip(mlp.intercepts_, skmlp.intercepts_):
            assert mi.shape == skmi.shape
            # assert_allclose(mi, skmi)

        assert len(mlp.intercepts_) == len(skmlp.intercepts_)
        assert_equal(mlp.classes_, skmlp.classes_)
        assert predict_mlp.shape == predict_skmlp.shape
        assert_allclose(predict_mlp, predict_skmlp, atol=atol)
        assert np.abs(score_skmlp - score) / np.abs(score_skmlp) < 0.02

    @repeat(10)
    def test_2(self):
        """
        Test the Multilayer Perceptron implementation in a bigger dataset and then
        compares it to the Scikit-Learn implementation.
        """
        X, y = generate_classification_dataset(
            n_samples=20000, n_features=10, n_classes=2
        )

        mlp = MLPClassifier(loss_function="bce", hidden_layer_sizes=(32, 64, 128))
        skmlp = SkMLP(
            hidden_layer_sizes=(32, 64, 128),
            solver="sgd",
            early_stopping=False,
            n_iter_no_change=200,
        )

        skmlp.fit(X, y)
        mlp.fit(X, y)

        predict_skmlp = skmlp.predict(X)
        predict_mlp = mlp.predict(X).squeeze()

        score_skmlp = skmlp.score(X, y)
        score = mlp.score(X, y)

        atol = math.floor(y.shape[0] * 0.02)

        assert mlp.n_features_in_ == skmlp.n_features_in_
        assert mlp.n_layers_ == skmlp.n_layers_
        assert mlp.n_outputs_ == skmlp.n_outputs_

        if mlp.out_activation_ == "sigmoid":
            # sklearn refers the "sigmoid" activatio function as "logistic"
            assert "logistic" == skmlp.out_activation_

        assert len(mlp.coefs_) == len(skmlp.coefs_)

        for mc, skmc in zip(mlp.coefs_, skmlp.coefs_):
            assert mc.shape == skmc.shape
            # assert_allclose(mc, skmc)

        for mi, skmi in zip(mlp.intercepts_, skmlp.intercepts_):
            assert mi.shape == skmi.shape
            # assert_allclose(mi, skmi)

        assert len(mlp.intercepts_) == len(skmlp.intercepts_)
        assert_equal(mlp.classes_, skmlp.classes_)
        assert predict_mlp.shape == predict_skmlp.shape
        assert_allclose(predict_mlp, predict_skmlp, atol=atol)
        assert np.abs(score_skmlp - score) / np.abs(score_skmlp) < 0.02
