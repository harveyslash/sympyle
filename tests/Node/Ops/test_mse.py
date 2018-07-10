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
        np_inputs = np.random.randn(1)
        np_targets = np.random.randn(1)
        eps = 0.000001
        error_tolerance = 0.0000000001

        inputs_min_eps = Tensor(np_inputs - eps)
        targets_min_eps = Tensor(np_targets - eps)
        mse_min_eps = MSE(inputs_min_eps, targets_min_eps)
        y_min_eps = mse_min_eps.forward()

        inputs_plus_eps = Tensor(np_inputs + eps)
        targets_plus_eps = Tensor(np_targets + eps)
        mse_plus_eps = MSE(inputs_plus_eps, targets_plus_eps)
        y_plus_eps = mse_plus_eps.forward()

        assert (np.abs(y_min_eps - y_plus_eps) < error_tolerance)


if __name__ == "__main__":
    unittest.main()
