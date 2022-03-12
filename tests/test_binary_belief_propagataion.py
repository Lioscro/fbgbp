from unittest import TestCase

import numpy as np

from fbgbp import binary_belief_propagation

from .mixins import TestMixin


class TestBinaryBeliefPropagation(TestMixin, TestCase):
    def test_init(self):
        with self.assertRaises(binary_belief_propagation.BeliefPropagationError):
            binary_belief_propagation.FastBinaryBeliefPropagation(
                4,
                np.array([2, 2, 2, 2], dtype=np.uint8),
                np.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.uint64),
                np.full(2, 0.9),
                np.full(4, 0.1),
                0.9, 0.1
            )

        with self.assertRaises(binary_belief_propagation.BeliefPropagationError):
            binary_belief_propagation.FastBinaryBeliefPropagation(
                4,
                np.array([2, 2, 2, 2], dtype=np.uint8),
                np.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.uint64),
                np.full(4, 0.9),
                np.full(2, 0.1),
                0.9, 0.1
            )

        with self.assertRaises(binary_belief_propagation.BeliefPropagationError):
            binary_belief_propagation.FastBinaryBeliefPropagation(
                8,
                np.array([2, 2, 2, 2], dtype=np.uint8),
                np.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.uint64),
                np.full(8, 0.9),
                np.full(8, 0.1),
                0.9, 0.1
            )

    def test_simple(self):
        bp = binary_belief_propagation.FastBinaryBeliefPropagation(
            4,
            np.array([2, 2, 2, 2], dtype=np.uint8),
            np.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.uint64),
            np.full(4, 0.9),
            np.full(4, 0.1),
            0.9, 0.1
        )
        bp.run()
        marginals = bp.marginals()
        np.testing.assert_allclose([1e-3, 1e-3, 1e-3, 1e-3],
                                   marginals,
                                   rtol=0,
                                   atol=0.005)

    def test_complex(self):
        shape = np.load(self.shape_path)
        n_nodes = np.prod(shape)
        n_neighbors = np.load(self.n_neighbors_path)
        neighbors = np.load(self.neighbors_path)
        potentials0 = np.load(self.potentials0_path)
        potentials1 = np.load(self.potentials1_path)
        marginals = np.load(self.marginals_path)
        bp = binary_belief_propagation.FastBinaryBeliefPropagation(
            n_nodes,
            n_neighbors,
            neighbors,
            potentials0,
            potentials1,
            0.7,
            0.3
        )
        bp.run(precision=1e-3, max_iter=100)
        np.testing.assert_allclose(marginals, bp.marginals().reshape(shape), atol=1e-2)
