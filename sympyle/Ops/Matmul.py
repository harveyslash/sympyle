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

    def _gradients_for_a(self, parent_grads):
        child_forward = self.children[1].forward()
        return np.dot(parent_grads,
                      child_forward.T)

    def _gradients_for_b(self, parent_grads):
        child_forward = self.children[0].forward()
        child_forward = child_forward.T

        return np.dot(child_forward,
                      parent_grads)

    def backward(self, respect_to_node, parent_grads=None):
        if parent_grads is None:
            raise AttributeError("MatMul cannot be a root op")

        if respect_to_node == self.children[0]:  # with respect to a
            return self._gradients_for_a(parent_grads)

        elif respect_to_node == self.children[1]:  # with respect to b
            return self._gradients_for_b(parent_grads)
