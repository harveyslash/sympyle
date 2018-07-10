import unittest
from Nodes import Add, Tensor
import numpy as np


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
        np_1 = np.random.randn(1)
        t1 = np.array(np_1)
        t1 = Tensor(t1)

        np_2 = np.random.randn(1)
        t2 = np.array(np_2)
        t2 = Tensor(t2)

        add_op = t1 + t2

        add_op.forward()

        assert add_op.forward_val == np_1 + np_2

    def test_vector_forward(self):
        """
        Test to see if element-wise vector forward values are computed correctly
        """
        np_1 = np.random.randn(10)
        t1 = np.array(np_1)
        t1 = Tensor(t1)

        np_2 = np.random.randn(10)
        t2 = np.array(np_2)
        t2 = Tensor(t2)

        add_op = t1 + t2

        add_op.forward()

        assert np.all(add_op.forward_val == (np_1 + np_2))

    def test_scalar_backward(self):
        np_1 = np.random.randn(1)
        t1 = np.array(np_1)
        t1 = Tensor(t1)

        np_2 = np.random.randn(1)
        t2 = np.array(np_2)
        t2 = Tensor(t2)

        add_op = t1 + t2

        add_op.forward()
        assert np.all(add_op.backward(t2) == np.array([1.0]))
        assert np.all(add_op.backward(t1) == np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
