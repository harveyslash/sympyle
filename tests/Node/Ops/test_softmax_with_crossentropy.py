import unittest

import numpy as np

from Nodes import SoftmaxWithCrossEntropy, Tensor
from .Helpers import calculate_numerical_gradient


class SoftmaxWithCrossEntropyOpTest(unittest.TestCase):

    def test_backward(self):
        """
        Compare gradient value using newton's method with the gradients
        computed by the graph.

        This test passes if the two values are within some error tolerance
        :return:
        """

        # Setup graph with random values
        np_inputs = np.linspace(0, 5, 5)
        np_targets = np.zeros(5)
        np_targets[0] = 1.0
        array_slice = slice(1)

        error_tolerance = 0.00000001

        inputs = Tensor(np_inputs)
        targets = Tensor(np_targets)
        entropy = SoftmaxWithCrossEntropy(inputs, targets, axis=0)
        entropy.backward()

        # compute derivatives from graph
        inputs_derivative = inputs.backward_val.copy()[array_slice]

        entropy.clear()
        inputs_numerical_deriv = calculate_numerical_gradient(entropy, inputs,
                                                              array_slice)

        assert np.all(np.abs(
                inputs_numerical_deriv - inputs_derivative) < error_tolerance)


if __name__ == "__main__":
    unittest.main()
