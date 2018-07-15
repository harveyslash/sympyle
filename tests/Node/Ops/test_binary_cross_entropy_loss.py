import unittest

import numpy as np

from sympyle import Tensor
from sympyle.Ops import BinaryCrossEntropyLoss, Sigmoid
from .Helpers import calculate_numerical_gradient


class TestBinaryCrossEntropyLoss(unittest.TestCase):

    def test_backward(self):
        """
        Compare gradient value using newton's method with the gradients
        computed by the graph.

        This test passes if the two values are within some error tolerance
        :return:
        """

        # Setup graph with random values
        np.random.seed(100)
        np_inputs = np.random.randn(1000, 1)
        np_targets = np.zeros((1000, 1))
        # np_targets[0] = 1.0
        array_slice = (0, 0)

        error_tolerance = 0.00000001

        inputs = Tensor(np_inputs)
        targets = Tensor(np_targets)
        sigmoid = Sigmoid(inputs)
        entropy = BinaryCrossEntropyLoss(sigmoid, targets, axis=1)
        entropy.forward()
        entropy.backward()

        # compute derivatives from graph
        inputs_derivative = inputs.backward_val.copy()[array_slice]

        entropy.clear()
        inputs_numerical_deriv = calculate_numerical_gradient(entropy, inputs,
                                                              array_slice)

        assert np.all(np.abs(
                inputs_numerical_deriv - inputs_derivative) < error_tolerance)
