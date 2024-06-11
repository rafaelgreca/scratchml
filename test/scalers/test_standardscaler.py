from numpy.testing import assert_almost_equal, assert_equal
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from ...scratchml.scalers import StandardScaler
from ..utils import generate_regression_dataset, repeat
import unittest


class Test_StandardScaler(unittest.TestCase):
    """
    Unittest class created to test the StandardScaler implementation.
    """

    @repeat(10)
    def test_1(self):
        """
        Test the StandarScaler technique using the default values
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset()

        skscaler = SkStandardScaler()
        scaler = StandardScaler()

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert_almost_equal(scaler.scale_, skscaler.scale_)
        assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert_almost_equal(scaler.var_, skscaler.var_)
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
        Test the StandarScaler technique without using the mean
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset()

        skscaler = SkStandardScaler(with_mean=False)
        scaler = StandardScaler(with_mean=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert_almost_equal(scaler.scale_, skscaler.scale_)
        # assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert_almost_equal(scaler.var_, skscaler.var_)
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
        Test the StandarScaler technique without using the standar deviation
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset()

        skscaler = SkStandardScaler(with_std=False)
        scaler = StandardScaler(with_std=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert scaler.scale_ == skscaler.scale_
        assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert scaler.var_ == skscaler.var_
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
        Test the StandarScaler technique without using the mean and standard deviation
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset()

        skscaler = SkStandardScaler(with_std=False, with_mean=False)
        scaler = StandardScaler(with_std=False, with_mean=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert scaler.scale_ == skscaler.scale_
        assert scaler.mean_ == skscaler.mean_
        assert scaler.var_ == skscaler.var_
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
        Test the StandarScaler technique using the default values on a higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkStandardScaler()
        scaler = StandardScaler()

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert_almost_equal(scaler.scale_, skscaler.scale_)
        assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert_almost_equal(scaler.var_, skscaler.var_)
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
        Test the StandarScaler technique without using the mean on a higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkStandardScaler(with_mean=False)
        scaler = StandardScaler(with_mean=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert_almost_equal(scaler.scale_, skscaler.scale_)
        # assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert_almost_equal(scaler.var_, skscaler.var_)
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
        Test the StandarScaler technique without using the standard deviation on a higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkStandardScaler(with_std=False)
        scaler = StandardScaler(with_std=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert scaler.scale_ == skscaler.scale_
        assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert scaler.var_ == skscaler.var_
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
        Test the StandarScaler technique without using the mean
        and standard deviation on a higher dimension and then compares it to
        the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=100)

        skscaler = SkStandardScaler(with_std=False, with_mean=False)
        scaler = StandardScaler(with_std=False, with_mean=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert scaler.scale_ == skscaler.scale_
        assert scaler.mean_ == skscaler.mean_
        assert scaler.var_ == skscaler.var_
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
        Test the StandarScaler technique using the default values on an even higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=1000)

        skscaler = SkStandardScaler()
        scaler = StandardScaler()

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert_almost_equal(scaler.scale_, skscaler.scale_)
        assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert_almost_equal(scaler.var_, skscaler.var_)
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
    def test_10(self):
        """
        Test the StandarScaler technique without using the mean on an even higher dimension
        and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=1000)

        skscaler = SkStandardScaler(with_mean=False)
        scaler = StandardScaler(with_mean=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert_almost_equal(scaler.scale_, skscaler.scale_)
        # assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert_almost_equal(scaler.var_, skscaler.var_)
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
    def test_11(self):
        """
        Test the StandarScaler technique without using the standard deviation
        on an even higher dimension and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=1000)

        skscaler = SkStandardScaler(with_std=False)
        scaler = StandardScaler(with_std=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert scaler.scale_ == skscaler.scale_
        assert_almost_equal(scaler.mean_, skscaler.mean_)
        assert scaler.var_ == skscaler.var_
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
    def test_12(self):
        """
        Test the StandarScaler technique without using the mean and the standard deviation
        on an even higher dimension and then compares it to the Scikit-Learn implementation.
        """
        X, _ = generate_regression_dataset(n_features=1000)

        skscaler = SkStandardScaler(with_std=False, with_mean=False)
        scaler = StandardScaler(with_std=False, with_mean=False)

        # fitting scalers
        skscaler.fit(X)
        scaler.fit(X)

        assert scaler.copy == skscaler.copy
        assert scaler.with_mean == skscaler.with_mean
        assert scaler.with_std == skscaler.with_std
        assert scaler.scale_ == skscaler.scale_
        assert scaler.mean_ == skscaler.mean_
        assert scaler.var_ == skscaler.var_
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
