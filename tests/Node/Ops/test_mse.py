import unittest
from Nodes import Add, Tensor, MSE
import numpy as np


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
        eps = 0.000001
        error_tolerance = 0.0000001

        inputs = Tensor(np_inputs)
        targets = Tensor(np_targets)
        mse = MSE(inputs, targets)
        y_min_eps = mse.forward()
        mse.backward()

        # compute derivatives from graph
        inputs_derivative = inputs.backward_val.copy()
        targets_derivative = targets.backward_val.copy()

        mse.clear()

        # modify inputs to inputs - eps
        inputs.value = np_inputs - eps
        targets.value = np_targets

        y_min_eps = mse.forward()

        mse.clear()

        # modify inputs to inputs + eps
        inputs.value = np_inputs + eps
        targets.value = np_targets
        y_plus_eps = mse.forward()

        mse.clear()

        numerical_deriv = (y_plus_eps - y_min_eps) / (2 * eps)

        assert (np.abs(numerical_deriv - inputs_derivative) < error_tolerance)

        # Compare derivatives of targets

        inputs.value = np_inputs
        # modify targets to targets - eps

        targets.value = np_targets - eps
        y_min_eps = mse.forward()

        mse.clear()

        inputs.value = np_inputs
        # modify targets to targets - eps
        targets.value = np_targets + eps
        y_plus_eps = mse.forward()

        numerical_deriv = (y_plus_eps - y_min_eps) / (2 * eps)

        assert (np.abs(numerical_deriv - targets_derivative) < error_tolerance)


if __name__ == "__main__":
    unittest.main()
