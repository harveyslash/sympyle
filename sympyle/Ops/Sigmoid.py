import numpy as np

from sympyle import Node


class Sigmoid(Node):
    """
    Element-wise sigmoid operation.
    """

    def __init__(self, input_tensor):
        super().__init__([input_tensor])

    def forward(self):
        """
        This performs sigmoid, which is given by
            1/(1 e^(-x)

        """
        child_forward = self.children[0].forward()
        return 1. / (1. + np.exp(-child_forward))

    def backward(self, respect_to_node, parent_grads=None):
        """
        the gradient of sigmoid is :
                sigmoid(x) * (1 - sigmoid(x))
        :param respect_to_node:
        :param parent_grads:
        :return:
        """
        forward_val = self.forward()

        if respect_to_node == self.children[0]:
            return parent_grads * forward_val * (1 - forward_val)
