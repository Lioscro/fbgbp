from unittest import TestCase

import numpy as np

from fbgbp import binary_grid_belief_propagation

from .mixins import TestMixin


class TestBinaryGridBeliefPropagation(TestMixin, TestCase):
    def test_init(self):
        with self.assertRaises(binary_grid_belief_propagation.BeliefPropagationError):
            binary_grid_belief_propagation.FastBinaryGridBeliefPropagation(
                np.array([2, 2], dtype=np.uint32),
                np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int16),
                np.full(2, 0.9),
                np.full(4, 0.1),
                0.9, 0.1
            )

        with self.assertRaises(binary_grid_belief_propagation.BeliefPropagationError):
            binary_grid_belief_propagation.FastBinaryGridBeliefPropagation(
                np.array([2, 2], dtype=np.uint32),
                np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int16),
                np.full(4, 0.9),
                np.full(2, 0.1),
                0.9, 0.1
            )

        with self.assertRaises(binary_grid_belief_propagation.BeliefPropagationError):
            binary_grid_belief_propagation.FastBinaryGridBeliefPropagation(
                np.array([2, 2, 2], dtype=np.uint32),
                np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int16),
                np.full(8, 0.9),
                np.full(8, 0.1),
                0.9, 0.1
            )

    def test_simple(self):
        shape = np.array([2, 2], dtype=np.uint32)
        neighbor_offsets = np.array([
            [-1, 0], [1, 0], [0, -1], [0, 1]
        ], dtype=np.int16)
        potentials0 = np.full(4, 0.9)
        potentials1 = np.full(4, 0.1)
        p = 0.9
        q = 0.1
        bp = binary_grid_belief_propagation.FastBinaryGridBeliefPropagation(
            shape, neighbor_offsets, potentials0, potentials1, p, q
        )
        bp.run()
        marginals = bp.marginals()
        np.testing.assert_allclose([[1e-3, 1e-3], [1e-3, 1e-3]],
                                   marginals,
                                   rtol=0,
                                   atol=0.005)

    def test_complex(self):
        shape = np.load(self.shape_path)
        potentials0 = np.load(self.potentials0_path)
        potentials1 = np.load(self.potentials1_path)
        neighbor_offsets = np.load(self.neighbor_offsets_path)
        marginals = np.load(self.marginals_path)
        bp = binary_grid_belief_propagation.FastBinaryGridBeliefPropagation(
            shape, neighbor_offsets, potentials0, potentials1, 0.7, 0.3
        )
        bp.run(precision=1e-3, max_iter=100)
        np.testing.assert_allclose(marginals, bp.marginals(), atol=1e-2)
