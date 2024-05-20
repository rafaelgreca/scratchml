import unittest
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from scratchml.preprocessing import StandardScaler
from test.utils import generate_regression_dataset, repeat

class Test_StandardScaler(unittest.TestCase):
    @repeat(10)
    def test_1(self):
        X, y = generate_regression_dataset()

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
        X, y = generate_regression_dataset()

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
        X, y = generate_regression_dataset()

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
        X, y = generate_regression_dataset()

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
        X, y = generate_regression_dataset(n_features=100)

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
        X, y = generate_regression_dataset(n_features=100)

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
        X, y = generate_regression_dataset(n_features=100)

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
        X, y = generate_regression_dataset(n_features=100)

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
        X, y = generate_regression_dataset(n_features=1000)

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
        X, y = generate_regression_dataset(n_features=1000)

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
        X, y = generate_regression_dataset(n_features=1000)

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
        X, y = generate_regression_dataset(n_features=1000)

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