import os
from unittest import TestCase


class TestMixin(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.fixtures_dir = os.path.join(cls.base_dir, 'fixtures')

        cls.marginals_path = os.path.join(cls.fixtures_dir, 'marginals.npy')
        cls.neighbor_offsets_path = os.path.join(cls.fixtures_dir, 'neighbor_offsets.npy')
        cls.potentials0_path = os.path.join(cls.fixtures_dir, 'potentials0.npy')
        cls.potentials1_path = os.path.join(cls.fixtures_dir, 'potentials1.npy')
        cls.shape_path = os.path.join(cls.fixtures_dir, 'shape.npy')

        cls.n_neighbors_path = os.path.join(cls.fixtures_dir, 'n_neighbors.npy')
        cls.neighbors_path = os.path.join(cls.fixtures_dir, 'neighbors.npy')
