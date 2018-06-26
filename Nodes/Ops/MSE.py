import numpy as np

from ..Node import Node


class MSE(Node):

    def __init__(self, y, t):
        super().__init__([y, t])

    @property
    def attributes(self):
        return {"color": 'blue', 'fill': "green"}
        pass

    def forward(self):
        y = self.children[0].forward()
        t = self.children[1].forward()

        avg = np.average((y - t) ** 2)
        return avg

    def _gradients_for_y(self):
        batch_size = self.children[0].forward().shape[0]
        y = self.children[0].forward()
        t = self.children[1].forward()
        output = (2 / batch_size) * (y - t)
        return output

    def _gradients_for_t(self):
        batch_size = self.children[0].forward().shape[0]
        y = self.children[0].forward()
        t = self.children[1].forward()
        output = (2 / batch_size) * (t - y)
        return output

    def backward(self, respect_to_node, parent_grads=None):
        if respect_to_node == self.children[0]:
            return self._gradients_for_y()
        elif respect_to_node == self.children[1]:
            return self._gradients_for_t()
