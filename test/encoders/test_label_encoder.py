import unittest
import numpy as np
from numpy.testing import assert_equal
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
from scratchml.encoders import LabelEncoder


class Test_LabelEncoder(unittest.TestCase):
    def test_1(self):
        le = LabelEncoder()
        le.fit([1, 2, 2, 6])
        le_transform = le.transform([1, 1, 2, 6])
        le_itransform = le.inverse_transform([0, 0, 1, 2])

        skle = SkLabelEncoder()
        skle.fit([1, 2, 2, 6])
        skle_transform = skle.transform([1, 1, 2, 6])
        skle_itransform = skle.inverse_transform([0, 0, 1, 2])

        assert_equal(le.classes_, skle.classes_)
        assert type(le.classes_) == type(skle.classes_)
        assert_equal(le_transform, skle_transform)
        assert type(le_transform) == type(skle_transform)
        assert_equal(le_itransform, skle_itransform)
        assert type(le_itransform) == type(skle_itransform)

    def test_2(self):
        le = LabelEncoder()
        le.fit(["paris", "paris", "tokyo", "amsterdam"])
        le_transform = le.transform(["tokyo", "tokyo", "paris"])
        le_itransform = le.inverse_transform([2, 2, 1])

        skle = SkLabelEncoder()
        skle.fit(["paris", "paris", "tokyo", "amsterdam"])
        skle_transform = skle.transform(["tokyo", "tokyo", "paris"])
        skle_itransform = skle.inverse_transform([2, 2, 1])

        assert_equal(le.classes_, skle.classes_)
        assert type(le.classes_) == type(skle.classes_)
        assert_equal(le_transform, skle_transform)
        assert type(le_transform) == type(skle_transform)
        assert_equal(le_itransform, skle_itransform)
        assert type(le_itransform) == type(skle_itransform)


if __name__ == "__main__":
    unittest.main(verbosity=2)
