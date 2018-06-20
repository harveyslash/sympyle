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
        if respect_to_node == self.children[0] or respect_to_node == \
                self.children[1]:
            back = self.parent.backward(self)

            return back

        raise AssertionError(
                "node not a direct child, cant calculate with respect to")
