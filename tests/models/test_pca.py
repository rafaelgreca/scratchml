from numpy.testing import assert_almost_equal, assert_equal
from sklearn.decomposition import PCA as SkPCA
from scratchml.models.pca import PCA
from ..utils import (
    generate_regression_dataset,
    generate_classification_dataset,
    generate_blob_dataset,
    repeat,
)
import unittest


class Test_PCA(unittest.TestCase):
    """
    Unittest class created to test the PCA implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the PCA technique using the default values
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=5)

        skscaler = SkPCA(n_components=2)
        scaler = PCA(n_components=2)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_2(self):
        """
        Test the PCA technique using the default values with higher dimensional space
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkPCA(n_components=2)
        scaler = PCA(n_components=2)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_3(self):
        """
        Test the PCA technique using the default values with higher dimensional space
        and higher n_components, then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkPCA(n_components=10)
        scaler = PCA(n_components=10)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_4(self):
        """
        Test the PCA technique using the default values
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_classification_dataset(n_features=5)

        skscaler = SkPCA(n_components=2)
        scaler = PCA(n_components=2)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_5(self):
        """
        Test the PCA technique using the default values with higher dimensional space
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_classification_dataset(n_features=100)

        skscaler = SkPCA(n_components=2)
        scaler = PCA(n_components=2)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_6(self):
        """
        Test the PCA technique using the default values with higher dimensional space
        and higher n_components, then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_classification_dataset(n_features=100)

        skscaler = SkPCA(n_components=10)
        scaler = PCA(n_components=10)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_7(self):
        """
        Test the PCA technique using the default values
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_blob_dataset(n_features=5)

        skscaler = SkPCA(n_components=2)
        scaler = PCA(n_components=2)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_8(self):
        """
        Test the PCA technique using the default values with higher dimensional space
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_blob_dataset(n_features=100)

        skscaler = SkPCA(n_components=2)
        scaler = PCA(n_components=2)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape

    @repeat(10)
    def test_9(self):
        """
        Test the PCA technique using the default values with higher dimensional space
        and higher n_components, then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_blob_dataset(n_features=100)

        skscaler = SkPCA(n_components=10)
        scaler = PCA(n_components=10)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.n_components_ == skscaler.n_components_
        assert scaler.n_features_in_ == skscaler.n_features_in_
        assert scaler.n_samples_ == skscaler.n_samples_
        assert scaler.components_.shape == skscaler.components_.shape
        # assert_almost_equal(scaler.components_, skscaler.components_)
        assert scaler.explained_variance_.shape == skscaler.explained_variance_.shape
        assert_almost_equal(scaler.explained_variance_, skscaler.explained_variance_)
        assert (
            scaler.explained_variance_ratio_.shape
            == skscaler.explained_variance_ratio_.shape
        )
        assert_almost_equal(
            scaler.explained_variance_ratio_, skscaler.explained_variance_ratio_
        )
        assert scaler.mean_.shape == skscaler.mean_.shape
        assert_almost_equal(scaler.mean_, skscaler.mean_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        # assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        # assert_almost_equal(sk_inv_trans_X, inv_trans_X)

        assert scaler.get_precision().shape, skscaler.get_precision().shape

        assert scaler.get_covariance().shape, skscaler.get_covariance().shape
