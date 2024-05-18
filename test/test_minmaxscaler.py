import pytest
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from ..scratchml.preprocessing import MinMaxScaler

@pytest.mark.parametrize("execution_number", range(10))
def test_1(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=1,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_2(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=1,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_3(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=1,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_4(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=100,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_5(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=100,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_6(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=100,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_7(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=1000,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_8(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=1000,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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

@pytest.mark.parametrize("execution_number", range(10))
def test_9(execution_number):
    X, y = make_regression(
        n_samples=10000,
        n_features=1000,
        n_targets=1,
        shuffle=True,
        noise=15
    )

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