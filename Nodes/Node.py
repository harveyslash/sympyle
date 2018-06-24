"""
Abstract node class to act as a symbolic placeholder for
operations.

Maintaining a graph like structure makes backprop easy to track.
"""

__author__ = "Harshvardhan Gupta"
from abc import ABC, abstractmethod
from functools import wraps
import pygraphviz as pgv

import types

__all__ = ['Node']


class Node(ABC):

    @property
    def attributes(self):
        return {}

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

    def __matmul__(self, other):
        from .Ops import Matmul

        return Matmul(self, other)

    def construct_graph(self, file_name, graph=None, i=1):

        if graph is None:
            graph = pgv.AGraph(directed=True)
            graph.layout('dot')

        i_root = i
        i_child = i + 1
        for child in self.children:
            graph.add_node(i_root, label=self.__class__.__name__,
                           **self.attributes)
            graph.add_node(i_child, label=child.__class__.__name__,
                           **child.attributes)
            graph.add_edge(i_child, i_root, color='green')
            graph.add_edge(i_root, i_child, color='red')
            i_child = child.construct_graph(file_name, graph, i_child)

        if i == 1:
            graph.draw(file_name, prog='dot')

        return i_child


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
