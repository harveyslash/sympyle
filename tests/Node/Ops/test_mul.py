import unittest

import numpy as np

from sympyle import Tensor
from .Helpers import calculate_numerical_gradient
from sympyle.Ops import MSE


class Add_Op(unittest.TestCase):
    """
    Class for testing functionality of Addition operation
    """

    def test_broadcast_backward(self):
        """
        Test broadcast functionality.
        Arrays of different shapes are broadcast together.
        The values of the gradients are compared with the gradients
        from the slope formula.
        """

        np.random.seed(100)

        a = Tensor(np.random.randn(1, 10))
        b = Tensor(np.random.randn(10, 1))
        c = Tensor(np.random.randn(1, 1))

        mse = MSE(a @ b, c)

        a_idx = (0, 0)
        b_idx = (0, 0)

        forward_val = mse.forward()
        assert forward_val.shape == ()
        mse.backward()

        a_grad = a.backward_val
        b_grad = b.backward_val

        assert a.value.shape == a.backward_val.shape
        assert b.value.shape == b.backward_val.shape

        a_numeric_grad = calculate_numerical_gradient(mse, a, a_idx)
        b_numeric_grad = calculate_numerical_gradient(mse, b, b_idx)

        assert np.abs(a_grad[a_idx] - a_numeric_grad) < 0.000001
        assert np.abs(b_grad[b_idx] - b_numeric_grad) < 0.000001
