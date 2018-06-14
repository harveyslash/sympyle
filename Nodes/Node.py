"""
Abstract node class to act as a symbolic placeholder for
operations.

Maintaining a graph like structure makes backprop easy to track.
"""

__author__ = "Harshvardhan Gupta"
from abc import ABC, abstractmethod


class Node(ABC):

    def register_parent_node(self, parent):
        self.parent = parent

    def __init__(self, children: list("Node")):
        """
        Register the parent a children of current node.

        :param children: a list of Node subclasses
        """
        self.children = children
        self.parent = Node
        for child in children:
            child.register_parent_node(self)

    @abstractmethod
    def forward(self):
        """
        Perform the op
        :return: the value after performing the op
        """
        pass

    @abstractmethod
    def backward(self, respect_to_node):
        """
        Calculate gradients of op
        :param respect_to_node: Which node to get the derivative with respect
        to. This is important if the op has more than one inputs
        :return: the derivative matrix
        """
        pass
