cimport numpy as np
import numpy as np
from libc.stdint cimport uint8_t, uint64_t
from libcpp cimport bool

from .errors import BeliefPropagationError

cdef class _BinaryBeliefPropagation:
    """Abstract class that all belief propagation classes should inherit from.
    Implements the `run` and `marginals` function.
    """
    def __dealloc__(self):
        del self.bp

    def run(
        self,
        double precision=.1,
        int max_iter=100,
        double log_bound=50.,
        bool taylor_approximation=False,
        int n_threads=1
    ):
        """Run belief propagation.

        Args:
            precision: Stop iteration once the desired precision has been reached.
                Precision is defined as the l2-norm of the previous iteration of
                messages vs this iteration of messages.
            max_iter: Maximum number of iterations.
            log_bound: Clip the log-messages within [-log_bound, log_bound] to
                prevent under/over-flow.
            taylor_approximation: Use Taylor approximation to compute messages.
                May cause slight loss of accuracy.
            n_threads: Number of threads to use.
        """
        self.bp.run(precision, max_iter, log_bound, taylor_approximation, n_threads)

    def marginals(self, double log_bound=50.):
        """Compute the marginal probability of being in state 1 of all nodes.

        Returns:
            A 1-dimensional Numpy array of the same length as the number of
            nodes. This is essentially the flattened multi-dimensional array.
        """
        cdef np.ndarray[np.double_t, ndim=1] marginals = np.empty(self.n_nodes)
        self.bp.marginals(&marginals[0], log_bound)
        return marginals


cdef class FastBinaryBeliefPropagation(_BinaryBeliefPropagation):
    """Belief propagation on an arbitrary graph with binary variables.

    Args:
        n_nodes: Number of nodes in the graph.
        n_neighbors: Number of neighbors for each node. Dtype must be `np.uint8_t`.
        neighbors: A 1-dimensional flattened array of neighbors of each node.
            Dtype must be `np.uint64_t`.
        potentials0: 1D Numpy array of node potentials corresponding to state 0.
            This should be the array produced by calling `.flatten()` on the
            actual N-dimensional array.
        potentials1: 1D Numpy array of node potentials corresponding to state 1.
            This should be the array produced by calling `.flatten()` on the
            actual N-dimensional array.
        p: Edge potential when neighboring states are the same.
        q: Edge potential when neighboring states are different.
    """
    def __cinit__(
        self,
        uint64_t n_nodes,
        np.ndarray[uint8_t, ndim=1, mode='c'] n_neighbors,
        np.ndarray[uint64_t, ndim=1, mode='c'] neighbors,
        np.ndarray[double, ndim=1, mode='c'] potentials0,
        np.ndarray[double, ndim=1, mode='c'] potentials1,
        double p,
        double q
    ):
        self.n_nodes = n_nodes
        if self.n_nodes != potentials0.shape[0] or self.n_nodes != potentials1.shape[0]:
            raise BeliefPropagationError(
                f'Exactly {self.n_nodes} potentials must be provided.'
            )

        if n_neighbors.shape[0] != self.n_nodes:
            raise BeliefPropagationError(
                f'`n_neighbors` must have `n_nodes` elements'
            )
        if n_neighbors.sum() != neighbors.shape[0]:
            raise BeliefPropagationError(
                f'`neighbors` must have number of elements equal to the sum of '
                '`n_neighbors`'
            )

        try:
            self.bp = new BinaryBeliefPropagation(
                n_nodes,
                &n_neighbors[0],
                &neighbors[0],
                &potentials0[0],
                &potentials1[0],
                p,
                q
            )
        except Exception as e:
            raise BeliefPropagationError(e)
