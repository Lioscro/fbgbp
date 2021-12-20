from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libcpp cimport bool

cdef extern from '../src/grid_belief_propagation.hpp':
    cdef cppclass GridBeliefPropagation:
        GridBeliefPropagation(uint8_t, const uint32_t*, const double*, const double*, double, double) except +
        void run(double, uint16_t, double, bool, uint64_t)
        void marginals(double*);
