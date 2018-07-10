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
        a = np.random.randn(1)
        t1 = Tensor(a)

        b = np.random.randn(1)
        t2 = Tensor(b)

        add_op = t1 + t2

        add_op.forward()

        assert add_op.forward_val == a + b

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

        assert np.all(add_op.forward_val == (a + b))

    def test_scalar_backward(self):
        """
            Test of add scalar backward 

            # Graph used to calculate expect values
        """

        a = np.random.randn(1)
        t1 = Tensor(a)
        b = np.random.randn(1)
        t2 = Tensor(b)

        add_op = t1 + t2
        add_op.forward()

        eps = 0.000001 # epsilon
        error_tolerance = 0.000001

        ## wrt `a`
        # x+eps
        input_eps = Tensor(a + eps)
        y_inp_eps_plus = input_eps + t2
        y_inp_eps_plus.forward()
        
        # x-eps
        input_eps.clear()
        input_eps.value = a - eps
        y_inp_eps_minus = input_eps + t2
        y_inp_eps_minus.forward()

        deriv_a = np.array((y_inp_eps_plus.forward_val - y_inp_eps_minus.forward_val) / (2 * eps))

        ## wrt `b`
        # x + eps
        input_eps.clear()
        input_eps.value = b + eps
        y_inp_eps_plus = input_eps + t1
        y_inp_eps_plus.forward()
        
        #x - eps
        input_eps.clear()
        input_eps.value = b - eps
        y_inp_eps_minus = input_eps + t1
        y_inp_eps_minus.forward()

        deriv_b = np.array((y_inp_eps_plus.forward_val - y_inp_eps_minus.forward_val) / (2 * eps))

        assert(np.abs(deriv_a - add_op.backward(t1)) < error_tolerance) # wrt a
        assert(np.abs(deriv_b - add_op.backward(t2)) < error_tolerance) # wrt b

    def test_scalar_backward_manual(self):
        """ 
            Manual test of add scalar backward
            i.e Graph not used to calculate expected value
        """
        a = np.random.randn(1)
        t1 = Tensor(a)

        b = np.random.randn(1)
        t2 = Tensor(b)

        add_op = t1 + t2
        add_op.forward()

        ## Expected output calculation
        # Using Newtons formular of (y(x+eps) - y(x-eps))/2*eps
        # `deriv_a` is the expected derivation with respect to `a`
        # `deriv_b` is the expected derivation with respect to `b`
        
        eps = 0.000001 # epsilon
        error_tolerance = 0.000001
        deriv_a = np.array((np.add(a + eps, b) - np.add(a - eps, b)) / (2 * eps)) 
        deriv_b = np.array((np.add(b + eps, a) - np.add(b - eps, a)) / (2 * eps))
        
        add_op.backward(t1)

        assert(np.abs(deriv_a - add_op.backward(t1)) < error_tolerance) # wrt a
        assert(np.abs(deriv_b - add_op.backward(t2)) < error_tolerance) # wrt b


if __name__ == "__main__":
    unittest.main()
