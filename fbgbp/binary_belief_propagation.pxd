from .wrappers.binary_belief_propagation cimport BinaryBeliefPropagation


cdef class _BinaryBeliefPropagation:
    cdef BinaryBeliefPropagation* bp
    cdef int n_nodes
