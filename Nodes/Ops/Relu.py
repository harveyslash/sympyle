import numpy as np

from ..Node import Node


class Relu(Node):
    def __init__(self, a):
        super().__init__([a])

    def forward(self):
        maxes = np.maximum(self.children[0].forward(), 0)
        return maxes

    def backward(self, respect_to_node):
        parent_grad = self.parent.backward(self).copy()
        maxes = self.forward()
        parent_grad[maxes <= 0] = 0
        # parent_grad[maxes > 0] = 1
        return parent_grad
