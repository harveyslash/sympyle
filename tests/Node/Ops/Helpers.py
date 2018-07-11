"""
Group of helper functions that are useful for tests
"""

__author__ = "Harshvardhan Gupta"
import numpy as np


def calculate_numerical_gradient(node, respect_to_node, node_slice=None,
                                 eps=0.000001):
    """
    Calculate gradient numerically using the formula:

            grad = (y(x-eps) - y(x+eps)) / 2*eps

    where eps is an infinitesimally small value.

    :param node: The output of this node will be used as y()
    :param respect_to_node: The node that represents x. This
                            Node's value changes by epsilon

    :param node_slice: In case of a respect to node that is non scalar,
                    a slice() object may be passed to modify the values that
                    are indexed by this object. The values indexed by this
                    parameter will be increased and decreased according to
                    eps

    :param eps:     The epsilon value to use
    :return:        the numerically computed gradient value
    """
    sliced = respect_to_node.value[node_slice]

    if len(sliced) > 1:
        raise AttributeError("Slice should result in a single value")

    node.clear()
    respect_to_node.value[node_slice] += eps
    plus_eps = node.forward()

    node.clear()
    respect_to_node.value[node_slice] -= 2 * eps
    minus_eps = node.forward()
    node.clear()

    respect_to_node.value[node_slice] += eps

    return (plus_eps - minus_eps) / (2 * eps)
