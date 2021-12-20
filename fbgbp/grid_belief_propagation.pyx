cimport cython
cimport numpy as np
import numpy as np
from libcpp cimport bool

from .wrappers.grid_belief_propagation cimport GridBeliefPropagation

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
    cdef np.ndarray shape
    cdef int n_nodes

    def __cinit__(
        self,
        np.ndarray[np.uint32_t, ndim=1] shape,
        np.ndarray[np.double_t, ndim=1] potentials0,
        np.ndarray[np.double_t, ndim=1] potentials1,
        double p,
        double q
    ):
        self.shape = shape
        self.n_nodes = np.prod(shape)
        if self.n_nodes != potentials0.shape[0] or self.n_nodes != potentials1.shape[0]:
            raise ValueError(f'Exactly {self.n_nodes} potentials must be provided.')

        potentials0 = np.ascontiguousarray(potentials0)
        potentials1 = np.ascontiguousarray(potentials1)

        self.bp = new GridBeliefPropagation(
            shape.shape[0], &shape[0], &potentials0[0], &potentials1[0], p, q
        )

    def __init__(self, shape: np.ndarray, potentials0: np.ndarray, potentials1: np.ndarray, p: float, q: float):
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
