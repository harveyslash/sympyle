import numpy as np

from Nodes.Node import consts
from ..Node import Node


class SoftmaxWithCrossEntropy(Node):
    """
    """

    def __init__(self, output, targets, axis=1):
        super().__init__([output, targets])
        self.axis = axis
        self.batch_axis = tuple(range(0, axis))
        self.softmax = None

    def forward(self):
        inputs = self.children[0].forward()
        targets = self.children[1].forward()

        exp = np.exp(inputs - inputs.max())
        summed = np.sum(exp, axis=self.axis, keepdims=True)

        softmax = exp / summed

        cross_entropies = - targets * np.log(softmax)

        cross_entropies = np.sum(cross_entropies, axis=self.axis)
        self.softmax = softmax

        if self.batch_axis:
            return np.average(cross_entropies, axis=self.batch_axis)

        return cross_entropies

    def backward(self, respect_to_node, parent_grads=1.0, **kwargs):
        self.forward()
        if respect_to_node == self.children[0]:
            softmax_output = self.softmax
            targets = self.children[1].forward()
            return (softmax_output - targets) / np.prod(self.batch_axis)

        elif respect_to_node == self.children[1]:
            return consts.no_grads
