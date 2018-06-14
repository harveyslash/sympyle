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
            return np.matmul(self.parent.backward(self),
                             self.children[1].forward().T)

        else:  # with respect to b
            return np.matmul(self.children[0].forward().T,
                             self.parent.backward(self))
