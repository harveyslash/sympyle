import numpy as np

from ..Node import Node


class MSE(Node):

    def __init__(self, y, t):
        super().__init__([y, t])

    def forward(self):
        y = self.children[0].forward()
        t = self.children[1].forward()

        avg = np.average((y - t) ** 2)
        return avg

    def backward(self, respect_to_node):
        batch_size = self.children[0].forward().shape[0]

        if respect_to_node == self.children[0]:
            y = self.children[0].forward()
            t = self.children[1].forward()
            output = (2 / batch_size) * (y - t)
            return output
        elif respect_to_node == self.children[1]:
            y = self.children[0].forward()
            t = self.children[1].forward()
            output = (2 / batch_size) * (t - y)
            return output

        raise AssertionError(
                "node not a direct child, cant calculate with respect to")
