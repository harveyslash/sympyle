from sympyle import Node
from sympyle.Ops import Add


class Sub(Node):
    """
    Sub op.
    """

    def __init__(self, a, b):
        super().__init__([a, b])
        self.custom_add_node = Add(a, b)

    def forward(self):
        return self.children[0].forward() - self.children[1].forward()

    def backward(self, respect_to_node, parent_grads=None, **kwargs):

        if respect_to_node == self.custom_add_node.children[0]:
            output = self.custom_add_node.backward(respect_to_node,
                                                   parent_grads,
                                                   should_save=False)
            return output
        if respect_to_node == self.custom_add_node.children[1]:
            output = -1 * self.custom_add_node.backward(respect_to_node,
                                                        parent_grads,
                                                        should_save=False)
            return output
