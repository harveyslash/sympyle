from ..Node import Node
from Nodes.Node import consts
import numpy as np


class SoftmaxWithCrossEntropy(Node):
    """
    Add op.
    """

    def __init__(self, output, targets, axis=1):
        super().__init__([output, targets])
        self.axis = axis

    def forward(self):
        inputs = self.children[0].forward()
        targets = self.children[1].forward()

        exp = np.exp(inputs)
        summed = np.sum(exp, axis=self.axis, keepdims=True)

        softmax = exp / summed

        cross_entropies = - targets * np.log(softmax)

        cross_entropies = np.sum(cross_entropies, axis=1)
        self.softmax = softmax
        # print("starting")
        # print(targets)
        # print(softmax)
        # print("done")
        # print(cross_entropies.sum())

        return np.average(cross_entropies)

    def backward(self, respect_to_node, parent_grads=1.0, **kwargs):
        if respect_to_node == self.children[0]:
            softmax_output = self.softmax
            targets = self.children[1].forward()
            return (softmax_output - targets) / softmax_output.shape[0]
        elif respect_to_node == self.children[1]:
            return consts.no_grads
