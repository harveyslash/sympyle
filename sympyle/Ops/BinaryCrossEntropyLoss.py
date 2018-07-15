import numpy as np

from sympyle import Node
from sympyle.Node import consts


class BinaryCrossEntropyLoss(Node):
    """
    This op fuses softmax and cross entropy together.
    Doing this is numerically more stable than the ops separately.
    """

    def __init__(self, output, targets, axis=1):

        super().__init__([output, targets])
        self.axis = axis
        self.batch_axis = tuple(range(0, axis))

    def forward(self):
        inputs = self.children[0].forward()
        targets = self.children[1].forward()

        bce = -(targets * np.log(inputs) + (1 - targets) * np.log(1 - inputs))
        return np.average(bce, self.batch_axis)

    def backward(self, respect_to_node, parent_grads=1.0, **kwargs):

        outputs = self.children[0].forward()
        targets = self.children[1].forward()

        if respect_to_node == self.children[0]:
            grads = - (targets / outputs) + (1 - targets) / (1 - outputs)

            return grads / np.prod(outputs.shape[0:self.axis])

        elif respect_to_node == self.children[1]:
            return consts.no_grads
