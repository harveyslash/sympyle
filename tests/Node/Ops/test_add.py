import unittest
from Nodes import Add, Tensor
import numpy as np
from .Helpers import calculate_numerical_gradient


class Add_Op(unittest.TestCase):
    """
    Class for testing functionality of Addition operation
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scalar_forward(self):
        """
        Test to see if scalar forward values are computed correctly
        """
        a = np.random.randn(1)
        t1 = Tensor(a)

        b = np.random.randn(1)
        t2 = Tensor(b)

        add_op = t1 + t2

        add_op.forward()

        e = f'''
        Test Failed!: test_scalar_forward()
        add_op.forward_val = {add_op.forward_val}, 
        (a+b) = {(a+b)}'''

        assert add_op.forward_val == (a + b), e

    def test_vector_forward(self):
        """
        Test to see if element-wise vector forward values are computed correctly
        """
        a = np.random.randn(10)
        t1 = Tensor(a)

        b = np.random.randn(10)
        t2 = Tensor(b)

        add_op = t1 + t2

        add_op.forward()

        e = f'''
        Test Failed!: test_vector_forward()
        add_op.forward_val = {add_op.forward_val}, 
        (a+b) = {(a+b)}'''

        assert np.all(add_op.forward_val == (a + b)), e

    def test_scalar_backward(self):
        """
            Test of add scalar backward 

            # Sympyle Graph used to calculate expect values
        """

        a = Tensor(np.random.rand(1))
        b = Tensor(np.random.rand(1))
        add_op = a + b
        add_op.backward()

        t1_grad = a.backward_val
        t2_grad = b.backward_val

        t1_numeric_grad = calculate_numerical_gradient(add_op, a,
                                                       slice(0, None))
        t2_numeric_grad = calculate_numerical_gradient(add_op, b,
                                                       slice(0, None))

        assert np.abs(t1_grad - t1_numeric_grad) < 0.00001
        assert np.abs(t2_grad - t2_numeric_grad) < 0.00001


if __name__ == "__main__":
    unittest.main()
