from ..Node import Node
import numpy as np



class Add(Node):
    """
    Add op.
    """

    def __init__(self, a, b):
        super().__init__([a, b])

    def forward(self):
        return np.add(self.children[0].forward(),
                         self.children[1].forward())

    def backward(self, respect_to_node):
        if respect_to_node == self.children[0] or respect_to_node == self.children[1]:  # with respect to a
            return self.parent.backward(self)

        raise AssertionError("node not a direct child, cant calculate with respect to")
