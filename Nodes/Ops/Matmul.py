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

    def backward(self, respect_to_node):
        if respect_to_node == self.children[0]:  # with respect to a
            parent_back = self.parent.backward(self)
            child_forward = self.children[1].forward()
            return np.dot(parent_back,
                          child_forward.T)

        elif respect_to_node == self.children[1]:  # with respect to b
            parent_back = self.parent.backward(self)
            child_forward = self.children[0].forward()
            child_forward = child_forward.T

            return np.dot(child_forward,
                          parent_back)

        raise AssertionError(
                "node not a direct child, cant calculate with respect to")
