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
        self.parent = None
        self.forward_val = None
        self.backward_val = None

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
    def backward(self, respect_to_node, parent_grads):
        """
        Calculate gradients of op
        :param respect_to_node: Which node to get the derivative with respect
        to. This is especially important if the op has more than one inputs
        :return: the derivative tensor
        """
        pass

    def __matmul__(self, other):
        from .Ops import Matmul

        return Matmul(self, other)

    def __add__(self, other):
        from .Ops import Add

        return Add(self, other)

    def __sub__(self, other):
        from .Ops import Sub

        return Sub(self, other)

    def draw_graph(self, file_name, graph=None, i=1):
        """
        Draw the computation graph visualisation and save it with the specified
        filename. This function recursively goes to all children and adds nodes
        to the graph object.

        :param file_name: the file name to save the visualisation on
        :param graph: internal parameter that should not be used externally
        :param i:     internal parameter that should not be used externally
        :return:      None
        """

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
            graph.add_edge(i_root, i_child, color='red', style='dashed')
            i_child = child.draw_graph(file_name, graph, i_child)

        if i == 1:
            graph.draw(file_name, prog='dot')

        return i_child

    def draw_graph(self, file_name, graph=None, root=True):
        """
        Draw the computation graph visualisation and save it with the specified
        filename. This function recursively goes to all children and adds nodes
        to the graph object.

        :param file_name: the file name to save the visualisation on
        :param graph: internal parameter that should not be used externally
        :param i:     internal parameter that should not be used externally
        :return:      None
        """

        if graph is None:
            graph = pgv.AGraph(directed=True)
            graph.layout('dot')

        graph.add_node(id(self), label=self.__class__.__name__,
                       **self.attributes)

        for child in self.children:
            graph.add_node(id(child), label=child.__class__.__name__,
                           **child.attributes)
            graph.add_edge(id(child), id(self), color='green')
            graph.add_edge(id(self), id(child), color='red', style='dashed')
            child.draw_graph(file_name, graph, False)

        if root:
            graph.draw(file_name, prog='dot')

        return None

    def clear(self):
        """
        Clear cached forward and backward values for all ops
        that are children of current op.

        :return: None
        """
        self.backward_val = None
        self.forward_val = None

        for child in self.children:
            child.clear()


def forward_decorator(func):
    """
    This decorates the forward function of a Node instance.
    This adds functionality to all node instances' forward() instances.

    Most importantly, if the forward value has already been computed, it
    will be reused.

    :param func: The function to be patched (see Node.py's init method)
    :return: The patched function with added functionality.
    """

    @wraps(func)
    def wrapper(s):
        if s.forward_val is not None:
            return s.forward_val

        s.forward_val = func()

        return s.forward_val

    return wrapper


def calculate_with_respect_to(respect_to_node, func, parent_grads,
                              should_save=True):
    """
    Calculate the value of a gradients of a certain node
    with respect to another node.

    This also checks if the gradient has already been computed. If it has,
    then it is reused instead of recomputation.


    After the gradient is computed for the child node , it is saved.
    Note that it is saved in the CHILD's backward_val and not the parent's.

    :param respect_to_node: The node to calculate the gradient with respect to.
            This node must be a direct child of the parent node.

    :param func: The function that will return the gradients for the
            respect_to_node.
    :param parent_grads: The gradients from the parent node.
    :param should_save:  If the child node's .backward_val should be updated
                        or not. This should be set to False when using ops
                        as part of other ops

    :return:
    """

    backward_val = func(respect_to_node, parent_grads)

    if backward_val is None:
        raise AssertionError(
                "node not a direct child, cant calculate with respect to")

    if not should_save:
        return backward_val
    if respect_to_node.backward_val is not None:
        respect_to_node.backward_val += backward_val
    else:
        respect_to_node.backward_val = backward_val.copy()

    return backward_val


def backward_decorator(func):
    """
    This decorates the backward function of a Node instance.
    This adds functionality to all node instances' backward() instances.

    it adds the following functionality:
    * calling backward() on any node without arguments will recursively
        calculate gradients for all child nodes
    * all computed gradients are saved


    Also read the helper function calculate_with_respect_to()

    :param func: The function to be patched (see Node.py's init method)
    :return: The patched function with added functionality.
    """

    @wraps(func)
    def wrapper(s: Node, respect_to_node=None, parent_grads=None,
                should_save=True):

        if respect_to_node is None:
            for child in s.children:
                grads = calculate_with_respect_to(child, func, parent_grads)
                child.backward(parent_grads=grads)
        else:
            return calculate_with_respect_to(respect_to_node, func,
                                             parent_grads, should_save)

    return wrapper
