import unittest
from Nodes import Add, Tensor, MSE
import numpy as np
from .Helpers import calculate_numerical_gradient


class MseOpTest(unittest.TestCase):

    def test_backward(self):
        """
        Compare gradient value using newton's method with the gradients
        computed by the graph.

        This test passes if the two values are within some error tolerance
        :return:
        """

        # Setup graph with random values
        np_inputs = np.random.randn(1)
        np_targets = np.random.randn(1)
        error_tolerance = 0.00001

        inputs = Tensor(np_inputs)
        targets = Tensor(np_targets)
        mse = MSE(inputs, targets)
        mse.backward()

        # compute derivatives from graph
        inputs_derivative = inputs.backward_val.copy()
        targets_derivative = targets.backward_val.copy()

        mse.clear()
        inputs_numerical_deriv = calculate_numerical_gradient(mse, inputs)
        targets_numerical_deriv = calculate_numerical_gradient(mse, targets)

        assert (np.abs(
                inputs_numerical_deriv - inputs_derivative) < error_tolerance)
        assert (np.abs(
                targets_numerical_deriv - targets_derivative) < error_tolerance)


if __name__ == "__main__":
    unittest.main()
