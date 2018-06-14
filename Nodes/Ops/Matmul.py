from ..Node import Node
import numpy as np

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
            return np.matmul(parent_back,
                             child_forward.T)

        elif respect_to_node == self.children[1]:  # with respect to b
            return np.matmul(self.children[0].forward().T,
                             self.parent.backward(self))

        raise AssertionError("node not a direct child, cant calculate with respect to")
