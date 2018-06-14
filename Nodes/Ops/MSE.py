from ..Node import Node
import numpy as np


class MSE(Node):

    def __init__(self, y, t):
        super().__init__([y, t])

    def forward(self):
        y = self.children[0].forward()
        t = self.children[1].forward()

        avg = np.average((y - t) ** 2)
        return avg

    def backward(self, respect_to_node):
        if respect_to_node == self.children[0]:
            y = self.children[0].forward()
            t = self.children[1].forward()
            output = (2 / t.shape[0]) * (y - t)
            return output
        else:
            y = self.children[0].forward()
            t = self.children[1].forward()
            output = (2 / t.shape[0]) * (t - y)
            return output
