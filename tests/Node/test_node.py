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

        file_name = tempfile.gettempdir() + str(hash(os.times())) + ".png"

        o.draw_graph(file_name)
        assert os.path.exists(file_name)
