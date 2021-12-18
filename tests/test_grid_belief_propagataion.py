from unittest import TestCase

import numpy as np

from fbgbp import grid_belief_propagation


class TestGridBeliefPropagation(TestCase):

    def test_simple(self):
        shape = np.array([2, 2], dtype=np.uint32)
        potentials0 = np.full(4, 0.9)
        potentials1 = np.full(4, 0.1)
        p = 0.9
        q = 0.1
        bp = grid_belief_propagation.FastBinaryGridBeliefPropagation(
            shape, potentials0, potentials1, p, q
        )
        bp.run()
        marginals = bp.marginals()
        np.testing.assert_allclose([[1e-3, 1e-3], [1e-3, 1e-3]],
                                   marginals,
                                   rtol=0,
                                   atol=0.005)
