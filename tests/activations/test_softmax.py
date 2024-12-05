from scratchml.activations import softmax
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_Softmax(unittest.TestCase):
    """
    Unittest class created to test the Softmax activation function.
    """

    @repeat(3)
    def test1(self):
        """
        Test the Softmax function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = softmax(X)
        s_pytorch = torch.nn.functional.softmax(torch.from_numpy(X), dim=-1).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(3)
    def test2(self):
        """
        Test the Softmax derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = softmax(X.detach().numpy(), derivative=True)
        torch.nn.functional.softmax(X, dim=-1).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the Softmax derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = softmax(X.detach().numpy(), derivative=True)
        torch.nn.functional.softmax(X, dim=-1).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
