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
        right = right.reshape(-1, 1)
        add = np.add(left, right)
        print(right.shape)
        print(left.shape)
        print(add.shape)
        return add

    def backward(self, respect_to_node):
        if respect_to_node == self.children[0] or respect_to_node == \
                self.children[1]:
            back = self.parent.backward(self)
            left = self.children[0].forward()
            right = self.children[1].forward()
            if left.shape != right.shape:
                print("back.shape")
                print(back.shape)

                back = np.sum(back, axis=1, keepdims=True)

                print("back.shape")
                print(back.shape)

            return back

        raise AssertionError(
                "node not a direct child, cant calculate with respect to")
