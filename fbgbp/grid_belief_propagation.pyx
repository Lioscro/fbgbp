cimport numpy as np
import numpy as np

from .wrappers.grid_belief_propagation cimport GridBeliefPropagation

cdef class FastBinaryGridBeliefPropagation:
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

    def __dealloc__(self):
        del self.bp

    def run(
        self,
        double precision=.1,
        int max_iter=100,
        double approximation_threshold=100.,
        int n_threads=1
    ):
        self.bp.run(precision, max_iter, approximation_threshold, n_threads)

    def marginals(self):
        cdef np.ndarray[np.double_t, ndim=1] marginals = np.empty(self.n_nodes)
        self.bp.marginals(&marginals[0])
        return marginals.reshape(self.shape)
