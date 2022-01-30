cimport cython
cimport numpy as np
import numpy as np
from libc.stdint cimport int16_t, uint32_t
from libc.stdlib cimport malloc, free

from .binary_belief_propagation cimport _BinaryBeliefPropagation
from .errors import BeliefPropagationError
from .wrappers.binary_belief_propagation cimport BinaryBeliefPropagation


cdef class FastBinaryGridBeliefPropagation(_BinaryBeliefPropagation):
    """Belief propagation on a grid MRF with binary variables. Has support for
    any-dimensional MRF.

    Args:
        shape: Shape of the grid as a Numpy array. Note that the dtype of the
            array must be `np.uint32`.
        neighbor_offsets: 2D Numpy array where each row corresponds to a
            potential neighbor, and the values correspond to the offests of
            each dimension from a node to that neighbor. Dtype must be `np.int16_t`.
        potentials0: 1D Numpy array of node potentials corresponding to state 0.
            This should be the array produced by calling `.flatten()` on the
            actual N-dimensional array.
        potentials1: 1D Numpy array of node potentials corresponding to state 1.
            This should be the array produced by calling `.flatten()` on the
            actual N-dimensional array.
        p: Edge potential when neighboring states are the same.
        q: Edge potential when neighboring states are different.
    """
    cdef int n_dims
    cdef np.ndarray shape

    def __cinit__(
        self,
        np.ndarray[uint32_t, ndim=1, mode='c'] shape,
        np.ndarray[int16_t, ndim=2, mode='c'] neighbor_offsets,
        np.ndarray[double, ndim=1, mode='c'] potentials0,
        np.ndarray[double, ndim=1, mode='c'] potentials1,
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

            self.bp = new BinaryBeliefPropagation(
                shape.shape[0],
                &shape[0],
                max_neighbors,
                &neighbor_offsets_p[0],
                &potentials0[0],
                &potentials1[0],
                p,
                q
            )
        except Exception as e:
            raise BeliefPropagationError(e)
        finally:
            free(neighbor_offsets_p)

    def marginals(self):
        """Compute the marginal probability of being in state 1 of all nodes.

        Returns:
            A N-dimensional Numpy array of the same shape as the shape provided
            upon initialization of this object.
        """
        return super().marginals().reshape(self.shape)
