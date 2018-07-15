import unittest

import numpy as np
from sympyle.Utils import get_broadcast_axes


class GradientBroadcastingTests(unittest.TestCase):
    def test_same_dims(self):
        """
        When the two shapes are exactly the same,there should be no broadcast.
        :return:
        """
        a = np.random.randn(1)
        b = np.random.randn(1)

        broadcast_axes = get_broadcast_axes(a + b, a)
        assert sorted(broadcast_axes) == sorted(())

        broadcast_axes = get_broadcast_axes(a + b, b)
        assert sorted(broadcast_axes) == sorted(())

    def test_unequal_dims(self):
        """
        When one array has more dimensions than another, the shorter
        array should be broadcast into the extra dimensions.
        :return:
        """
        a = np.random.randn(1, 100)
        b = np.random.randn(1)

        broadcast_axes = get_broadcast_axes(a + b, a)
        assert sorted(broadcast_axes) == sorted(())

        broadcast_axes = get_broadcast_axes(a + b, b)
        assert sorted(broadcast_axes) == sorted((1, 0))

    def test_multiple_axes_broadcast(self):
        """
        When multiple axes can be broadcast, the function should return
        all the axes over which the array was broadcast.
        :return:
        """
        a = np.random.randn(1, 8, 1, 100)
        b = np.random.randn(3, 1, 8, 1)

        broadcast_axes = get_broadcast_axes(a + b, a)
        assert sorted(broadcast_axes) == sorted((0, 2))

        broadcast_axes = get_broadcast_axes(a + b, b)
        assert sorted(broadcast_axes) == sorted((1, 3))

    def test_multiple_axes_unequal_dims(self):
        """
        When arrays have compatible shapes but unequal dimensions,
        the smaller array is broadcast into the 'extra' axes.
        :return:
        """
        a = np.random.randn(1, 3, 1, 100)
        b = np.random.randn(3, 1)

        broadcast_axes = get_broadcast_axes(a + b, a)
        assert sorted(broadcast_axes) == sorted((2,))

        broadcast_axes = get_broadcast_axes(a + b, b)
        assert sorted(broadcast_axes) == sorted((0, 1, 3))
