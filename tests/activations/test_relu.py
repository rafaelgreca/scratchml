from scratchml.activations import relu
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_RELU(unittest.TestCase):
    """
    Unittest class created to test the RELU activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the RELU function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = relu(X)
        s_pytorch = torch.relu(torch.from_numpy(X)).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the RELU derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = relu(X.detach().numpy(), derivative=True)
        torch.relu(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the RELU derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = relu(X.detach().numpy(), derivative=True)
        torch.relu(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
