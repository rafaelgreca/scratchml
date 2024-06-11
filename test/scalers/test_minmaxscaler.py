from numpy.testing import assert_almost_equal, assert_equal
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from ...scratchml.scalers import MinMaxScaler
from ..utils import generate_regression_dataset, repeat
import unittest


class Test_MinMaxScaler(unittest.TestCase):
    """
    Unittest class created to test the MinMaxScaler implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the MinMaxScaler technique using the default values
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset()

        skscaler = SkMinMaxScaler()
        scaler = MinMaxScaler()

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_2(self):
        """
        Test the MinMaxScaler technique using clip
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset()

        skscaler = SkMinMaxScaler(clip=True)
        scaler = MinMaxScaler(clip=True)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_3(self):
        """
        Test the MinMaxScaler technique using a different feature_range
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset()

        skscaler = SkMinMaxScaler(feature_range=(-1, 1))
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_4(self):
        """
        Test the MinMaxScaler technique using the default values on a higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_samples=100)

        skscaler = SkMinMaxScaler()
        scaler = MinMaxScaler()

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_5(self):
        """
        Test the MinMaxScaler technique using clip on a higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkMinMaxScaler(clip=True)
        scaler = MinMaxScaler(clip=True)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_6(self):
        """
        Test the MinMaxScaler technique using a different feature range on a higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkMinMaxScaler(feature_range=(-1, 1))
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_7(self):
        """
        Test the MinMaxScaler technique using the default values on an even bigger dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=1000)

        skscaler = SkMinMaxScaler()
        scaler = MinMaxScaler()

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_8(self):
        """
        Test the MinMaxScaler technique using clip on an even bigger dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=1000)

        skscaler = SkMinMaxScaler(clip=True)
        scaler = MinMaxScaler(clip=True)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)

    @repeat(10)
    def test_9(self):
        """
        Test the MinMaxScaler technique using a different feature range on an even bigger dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=1000)

        skscaler = SkMinMaxScaler(feature_range=(-1, 1))
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.feature_range == skscaler.feature_range
        assert scaler.copy == skscaler.copy
        assert scaler.clip == skscaler.clip
        assert_equal(scaler.min_, skscaler.min_)
        assert_equal(scaler.scale_, skscaler.scale_)
        assert_equal(scaler.data_min_, skscaler.data_min_)
        assert_equal(scaler.data_max_, skscaler.data_max_)
        assert_equal(scaler.data_range_, skscaler.data_range_)
        assert_equal(scaler.n_features_in_, skscaler.n_features_in_)
        assert_equal(scaler.n_samples_seen_, skscaler.n_samples_seen_)

        # transforming the data
        sk_trans_X = skscaler.transform(X)
        trans_X = scaler.transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # fit transforming the data
        sk_trans_X = skscaler.fit_transform(X)
        trans_X = scaler.fit_transform(X)

        assert sk_trans_X.shape, trans_X.shape
        assert_almost_equal(sk_trans_X, trans_X)

        # inverse transform
        sk_inv_trans_X = skscaler.inverse_transform(sk_trans_X)
        inv_trans_X = scaler.inverse_transform(trans_X)

        assert sk_inv_trans_X.shape, inv_trans_X.shape
        assert_almost_equal(sk_inv_trans_X, inv_trans_X)


if __name__ == "__main__":
    unittest.main(verbosity=2)
