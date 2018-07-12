import os
import tempfile
import unittest

import numpy as np

from sympyle import Tensor


class NodeOp(unittest.TestCase):
    """
    Class for Generic Node functionality
    """

    def test_graph_draw(self):
        """
        Test graph drawing functionality.
        The test passes if the graph image was successfully created.
        """
        a = Tensor(np.random.randn(5))
        b = Tensor(np.random.randn(5))

        o = a + b

        fname = str(hash(os.times())) + ".png"
        full_name = os.path.join(tempfile.gettempdir(), fname)

        o.draw_graph(full_name)
        assert os.path.exists(full_name)
