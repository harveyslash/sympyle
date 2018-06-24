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

    def backward(self, respect_to_node):

        back = self.parent.backward(self)
        if respect_to_node == self.children[0]:
            child = self.children[0].forward()
        if respect_to_node == self.children[1]:
            child = self.children[1].forward()

        for i, (dim1, dim2) in enumerate(
                zip(child.shape[::-1], back.shape[::-1])):
            if dim2 != dim1:
                break
        if dim1 == dim2:
            return back
        return back.sum(axis=tuple(range(i)))
