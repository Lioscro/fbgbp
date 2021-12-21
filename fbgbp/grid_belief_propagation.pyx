cimport cython
cimport numpy as np
import numpy as np
from libc.stdint cimport int16_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool

from .wrappers.grid_belief_propagation cimport GridBeliefPropagation

class BeliefPropagationError(Exception): pass

cdef class FastBinaryGridBeliefPropagation:
    """Belief propagation on a grid MRF with binary variables. Has support for
    any-dimensional MRF.

    Args:
        shape: Shape of the grid as a Numpy array. Note that the dtype of the
            array must be `np.uint32`.
        potentials0: 1D Numpy array of node potentials corresponding to state 0.
            This should be the array produced by calling `.flatten()` on the
            actual N-dimensional array.
        potentials1: 1D Numpy array of node potentials corresponding to state 1.
            This should be the array produced by calling `.flatten()` on the
            actual N-dimensional array.
        p: Edge potential when neighboring states are the same.
        q: Edge potential when neighboring states are different.
    """
    cdef GridBeliefPropagation* bp
    cdef int n_dims
    cdef np.ndarray shape
    cdef int n_nodes

    def __cinit__(
        self,
        np.ndarray[np.uint32_t, ndim=1, mode='c'] shape,
        np.ndarray[np.int16_t, ndim=2, mode='c'] neighbor_offsets,
        np.ndarray[np.double_t, ndim=1, mode='c'] potentials0,
        np.ndarray[np.double_t, ndim=1, mode='c'] potentials1,
        double p,
        double q
    ):
        self.n_dims = shape.shape[0]
        self.shape = shape
        self.n_nodes = np.prod(shape)
        if self.n_nodes != potentials0.shape[0] or self.n_nodes != potentials1.shape[0]:
            raise BeliefPropagationError(
                f'Exactly {self.n_nodes} potentials must be provided.'
            )

        if neighbor_offsets.shape[1] != self.n_dims:
            raise BeliefPropagationError(
                f'`neighbor_offsets` must have {self.n_dims} as its second dimension.'
            )

        # Convert neighbor mask to C array
        cdef int max_neighbors = neighbor_offsets.shape[0]
        cdef int16_t** neighbor_offsets_p = <int16_t**>malloc(max_neighbors * sizeof(int16_t*))
        if not neighbor_offsets_p:
            raise MemoryError('Failed to allocate memory.')

        try:
            for i in range(max_neighbors):
                neighbor_offsets_p[i] = &neighbor_offsets[i, 0]
            potentials0 = np.ascontiguousarray(potentials0)
            potentials1 = np.ascontiguousarray(potentials1)

            self.bp = new GridBeliefPropagation(
                shape.shape[0],
                &shape[0],
                max_neighbors,
                &neighbor_offsets_p[0],
                &potentials0[0],
                &potentials1[0],
                p,
                q
            )
        finally:
            free(neighbor_offsets_p)

    def __init__(
        self,
        shape: np.ndarray,
        neighbor_offsets: np.ndarray,
        potentials0: np.ndarray,
        potentials1: np.ndarray,
        p: float,
        q: float
    ):
        pass

    def __dealloc__(self):
        del self.bp

    def run(
        self,
        double precision=.1,
        int max_iter=100,
        double log_bound=100.,
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

    def marginals(self):
        """Compute the marginal probability of being in state 1 of all nodes.

        Returns:
            An N-dimensional Numpy array in the same shape as the `shape`
            parameter provided when initializing the class.
        """
        cdef np.ndarray[np.double_t, ndim=1] marginals = np.empty(self.n_nodes)
        self.bp.marginals(&marginals[0])
        return marginals.reshape(self.shape)
