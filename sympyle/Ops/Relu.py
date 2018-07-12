import numpy as np

from sympyle import Node


class Relu(Node):
    def __init__(self, a):
        super().__init__([a])

    def forward(self):
        maxes = np.maximum(self.children[0].forward(), 0)
        return maxes

    def backward(self, respect_to_node, parent_grads=None):
        maxes = self.forward()
        parent_grads[maxes <= 0] = 0
        # parent_grad[maxes > 0] = 1
        return parent_grads
