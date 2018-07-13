import unittest

import numpy as np

from sympyle import Tensor
from sympyle.Ops import MSE, Sigmoid
from .Helpers import calculate_numerical_gradient


class Sigmoid_Op(unittest.TestCase):
    """
    Class for testing functionality of Sigmoid operation
    """

    def test_backward(self):
        """
        Test sigmoid gradient calculation.

        For simplifying testing, a graph with multiple nodes is used.
        if d(root)/d(x) is same from numerical as well as programmatic
        gradients, the test is passed.
        """

        inputs = Tensor(np.random.randn(10))
        targets = Tensor(np.random.randn(10))

        sigmoid = Sigmoid(inputs)

        mse = MSE(sigmoid, targets)

        a_idx = (0,)

        forward_val = mse.forward()
        assert forward_val.shape == ()
        mse.backward()

        a_grad = inputs.backward_val

        assert inputs.value.shape == inputs.backward_val.shape

        a_numeric_grad = calculate_numerical_gradient(mse, inputs, a_idx)

        diff = np.abs(a_grad[a_idx] - a_numeric_grad)
        assert diff < 0.0000001
