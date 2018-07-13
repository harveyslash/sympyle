import unittest

import numpy as np

from sympyle import Tensor
from .Helpers import calculate_numerical_gradient
from sympyle.Ops import Relu, MSE


class Relu_Op(unittest.TestCase):
    """
    Class for testing functionality of Relu operation
    """

    def test_backward(self):
        """
        Test Relu's gradient calculation.

        For simplifying testing, a graph with multiple nodes is used.
        if d(root)/d(x) is same from numerical as well as programmatic
        gradients, the test is passed.
        """

        a = Tensor(np.random.randn(10, 10))
        b = Tensor(np.random.randn(10, 1))
        c = Tensor(np.random.randn(10, 1))

        matmul = a @ b
        relu = Relu(matmul)

        mse = MSE(relu, c)

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
