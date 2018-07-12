import unittest

import numpy as np

from sympyle import Tensor
from .Helpers import calculate_numerical_gradient


class Add_Op(unittest.TestCase):
    """
    Class for testing functionality of Addition operation
    """

    def test_scalar_forward(self):
        """
        Test to see if scalar forward values are computed correctly
        """
        a = np.random.randn(1)
        t1 = Tensor(a)

        b = np.random.randn(1)
        t2 = Tensor(b)

        add_op = t1 - t2

        add_op.forward()

        assert add_op.forward_val == (a - b)

    def test_vector_forward(self):
        """
        Test to see if element-wise vector forward values are computed correctly
        """
        a = np.random.randn(10)
        t1 = Tensor(a)

        b = np.random.randn(10)
        t2 = Tensor(b)

        add_op = t1 - t2

        add_op.forward()

        assert np.all(add_op.forward_val == (a - b))

    def test_scalar_backward(self):
        """
            Test of add scalar backward 

            # Sympyle Graph used to calculate expect values
        """

        a = Tensor(np.random.rand(1))
        b = Tensor(np.random.rand(1))
        add_op = a - b
        add_op.backward()

        t1_grad = a.backward_val
        t2_grad = b.backward_val

        a_numeric_grad = calculate_numerical_gradient(add_op, a,
                                                      (0,))
        b_numeric_grad = calculate_numerical_gradient(add_op, b,
                                                      (0,))

        assert np.abs(t1_grad - a_numeric_grad) < 0.00001
        assert np.abs(t2_grad - b_numeric_grad) < 0.00001

    def test_broadcast_backward(self):
        """
        Test broadcast functionality.
        Arrays of different shapes are broadcast together.
        The values of the gradients are compared with the gradients
        from the slope formula.
        """

        np.random.seed(100)

        a = Tensor(np.random.randn(1, 1, 10, 3))
        b = Tensor(np.random.randn(1, 10, 3, 10, 1))

        a_idx = (0, 0, 1, 1)
        b_idx = (0, 1, 1, 1, 0)

        add_op = a - b

        forward_val = add_op.forward()
        assert forward_val.shape == (1, 10, 3, 10, 3)
        add_op.backward()

        a_grad = a.backward_val
        b_grad = b.backward_val

        assert a.value.shape == a.backward_val.shape
        assert b.value.shape == b.backward_val.shape

        a_numeric_grad = calculate_numerical_gradient(add_op, a, a_idx)
        b_numeric_grad = calculate_numerical_gradient(add_op, b, b_idx)

        assert np.abs(a_grad[a_idx] - a_numeric_grad) < 0.000001
        assert np.abs(b_grad[b_idx] - b_numeric_grad) < 0.000001
