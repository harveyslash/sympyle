from ..Node import Node
import numpy as np


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
            back = np.array([1.0])
            return back

        if respect_to_node == self.children[0]:
            child = self.children[0].forward()
        elif respect_to_node == self.children[1]:
            child = self.children[1].forward()

        dim1 = dim2 = 0
        for i, (dim1, dim2) in enumerate(
                zip(child.shape[::-1], back.shape[::-1])):
            if dim2 != dim1:
                break

        if dim1 == dim2:
            return back

        return back.sum(axis=tuple(range(i)))
