from scratchml.activations import leaky_relu
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_Leaky_RELU(unittest.TestCase):
    """
    Unittest class created to test the Leaky RELU activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the Leaky RELU function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = leaky_relu(X)
        s_pytorch = torch.nn.functional.leaky_relu(
            torch.from_numpy(X),
            negative_slope=0.001,
        ).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the Leaky RELU derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = leaky_relu(X.detach().numpy(), derivative=True)
        torch.nn.functional.leaky_relu(X, negative_slope=0.001).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the Leaky RELU derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = leaky_relu(X.detach().numpy(), derivative=True)
        torch.nn.functional.leaky_relu(X, negative_slope=0.001).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
