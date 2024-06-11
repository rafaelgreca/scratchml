from numpy.testing import assert_equal
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from ...scratchml.encoders import OneHotEncoder
import unittest


class Test_OneHotEncoder(unittest.TestCase):
    """
    Unittest class created to test the One Hot Encoder technique.
    """

    def test_1(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'infrequent_if_exist' with a custom max_categories value and
        then compares it to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=2)
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=2)
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_2(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'ignore' and then compares it to the Scikit-Learn implementation.
        """
        X = [["Male", 1], ["Female", 3], ["Female", 2]]
        test = [["Female", 1], ["Male", 4]]

        enc = SkOneHotEncoder(handle_unknown="ignore")
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_3(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'ignore' with 'if_binary' drop option and then compares it
        to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(handle_unknown="ignore", drop="if_binary")
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_4(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'ignore' with a custom drop option and then compares it
        to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(handle_unknown="ignore", drop=["Other", 1])
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="ignore", drop=["Other", 1])
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_5(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'infrequent_if_exists' with a custom min frequency and then compares it
        to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2)
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2)
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_6(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'infrequent_if_exists', a custom min frequency and max categories
        and then compares it to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(
            handle_unknown="infrequent_if_exist", max_categories=2, min_frequency=2
        )
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=2, max_categories=2
        )
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))


if __name__ == "__main__":
    unittest.main(verbosity=2)
