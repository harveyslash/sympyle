import numpy as np

from .. import Utils
from ..Node import Node


class Add(Node):
    """
    Add op.
    """

    def __init__(self, a, b):
        super().__init__([a, b])

    def forward(self):
        left = self.children[0].forward()
        right = self.children[1].forward()
        add = np.add(left, right)
        return add

    def backward(self, respect_to_node, parent_grads=None, **kwargs):

        back = parent_grads
        if back is None:
            back = np.ones_like(self.forward())

        if respect_to_node == self.children[0]:
            child = self.children[0].forward()
        elif respect_to_node == self.children[1]:
            child = self.children[1].forward()
        else:
            return None

        broadcast_dims = Utils.get_broadcast_axes(back, child)

        grad = back.sum(axis=broadcast_dims).reshape(child.shape)
        return grad
