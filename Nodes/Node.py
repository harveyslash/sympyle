"""
Abstract node class to act as a symbolic placeholder for
operations.

Maintaining a graph like structure makes backprop easy to track.
"""

__author__ = "Harshvardhan Gupta"
from abc import ABC, abstractmethod
from functools import wraps

import types

__all__ = ['Node']


class Node(ABC):

    def register_parent_node(self, parent):
        self.parent = parent

    def __init__(self, children: list("Node")):
        """
        Register the parent a children of current node.

        This function also wraps the forward and backward functions
        around a cached version of the functions.
        The first time forward() is called, the results are computed,
        but subsequent calls uses this computed values.

        To force recompute, self.backward_val and self.forward_val should be
        cleared.

        :param children: a list of Node subclasses
        """

        self.children = children
        self.parent = Node
        self.forward_val = None
        self.backward_val = None
        for child in children:
            child.register_parent_node(self)

        caching_func = forward_decorator(self.forward)
        self.forward = types.MethodType(caching_func, self)

        caching_func = backward_decorator(self.backward)
        self.backward = types.MethodType(caching_func, self)

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

    # def construct_graph(self, root, graph, i=1):
    #     if root == None:
    #         return i
    #     i_root = i
    #     i_child = i + 1
    #     for child in root.children:
    #         graph.add_node(i_root, label=root.__class__.__name__)
    #         graph.add_node(i_child, label=child.__class__.__name__)
    #         graph.add_edge(i_root, i_child)
    #         i_child = self.construct_graph(child, graph, i_child)
    #
    #     return i_child

    def clear_caches(self):
        self.backward_val = None
        self.forward_val = None

        for child in self.children:
            child.clear_caches()


def forward_decorator(func):
    @wraps(func)
    def wrapper(s):
        if s.forward_val is not None:
            return s.forward_val

        s.forward_val = func()

        return s.forward_val

    return wrapper


def calculate_with_respect_to(respect_to_node, func):
    if respect_to_node.backward_val is not None:
        return respect_to_node.backward_val

    respect_to_node.backward_val = func(respect_to_node)

    if respect_to_node.backward_val is None:
        raise AssertionError(
                "node not a direct child, cant calculate with respect to")

    return respect_to_node.backward_val


def backward_decorator(func):
    @wraps(func)
    def wrapper(s: Node, respect_to_node=None):

        if respect_to_node is None:
            for child in s.children:
                child.backward()
                calculate_with_respect_to(child, func)
        else:
            return calculate_with_respect_to(respect_to_node, func)

    return wrapper
