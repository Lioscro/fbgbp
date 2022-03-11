from libc.stdint cimport int16_t, uint8_t, uint16_t, uint32_t, uint64_t
from libcpp cimport bool

cdef extern from '../src/binary_belief_propagation.hpp':
    cdef cppclass BinaryBeliefPropagation:
        BinaryBeliefPropagation(uint8_t, const uint32_t*, uint8_t, int16_t**, const double*, const double*, double, double) except +
        BinaryBeliefPropagation(uint64_t, const uint8_t*, const uint64_t*, const double*, const double*, double, double) except +
        void run(double, uint16_t, double, bool, uint64_t)
        void marginals(double*, double);
