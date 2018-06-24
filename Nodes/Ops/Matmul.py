import numpy as np

from ..Node import Node

__author__ = "Harshvardhan Gupta"


class Matmul(Node):
    """
    Matrix multiplication op.
    """

    def __init__(self, a, b):
        super().__init__([a, b])

    def forward(self):

        return np.matmul(self.children[0].forward(),
                         self.children[1].forward())

    def _gradients_for_a(self):
        parent_back = self.parent.backward(self)
        child_forward = self.children[1].forward()
        return np.dot(parent_back,
                      child_forward.T)

    def _gradients_for_b(self):
        parent_back = self.parent.backward(self)
        child_forward = self.children[0].forward()
        child_forward = child_forward.T

        return np.dot(child_forward,
                      parent_back)

    def backward(self, respect_to_node):
        if respect_to_node == self.children[0]:  # with respect to a
            return self._gradients_for_a()

        elif respect_to_node == self.children[1]:  # with respect to b
            return self._gradients_for_b()
