import unittest

import numpy as np

from sympyle import Tensor
from .Helpers import calculate_numerical_gradient


class Sub_Op(unittest.TestCase):
    """
    Class for testing functionality of Subtraction operation
    """

    def test_broadcast_backward(self):
        """
        Test broadcast functionality.
        Arrays of different shapes are broadcast together.
        The values of the gradients are compared with the gradients
        from the slope formula.
        """

        a = Tensor(np.random.randn(1, 10, 3))
        b = Tensor(np.random.randn(4, 10, 3))

        a_idx = (0, 0, 1)
        b_idx = (0, 1, 1)

        add_op = a - b

        forward_val = add_op.forward()
        assert forward_val.shape == (4, 10, 3)
        add_op.backward()

        a_grad = a.backward_val
        b_grad = b.backward_val

        assert a.value.shape == a.backward_val.shape
        assert b.value.shape == b.backward_val.shape

        a_numeric_grad = calculate_numerical_gradient(add_op, a, a_idx)
        b_numeric_grad = calculate_numerical_gradient(add_op, b, b_idx)

        assert np.abs(a_grad[a_idx] - a_numeric_grad) < 0.000001
        assert np.abs(b_grad[b_idx] - b_numeric_grad) < 0.000001
