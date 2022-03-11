#ifndef BINARY_BELIEF_PROPAGATION_HPP_
#define BINARY_BELIEF_PROPAGATION_HPP_
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

class BinaryBeliefPropagation {
private:
    uint64_t n_nodes;

    // Variables defining edge/neighborhood structure.
    uint8_t* n_neighbors;
    uint64_t** neighbors;

    // Params
    float p;  // xi == xj
    float q;  // xi != xj
    float alpha; // p / q
    float log_alpha;  // log(p / q)

    // Messages -- messages are indexed in the order they appear in the
    // neighbors array. For example, for node i if we have neighbors[i][0] == j,
    // and this is the k'th element in the array, then messages[k] is the message
    // sent j -> i.
    uint64_t n_messages;
    float* messages;
    uint64_t* message_index;
    float* lambda;

    void initialize_alpha(double p, double q);
    void initialize_neighbors(const uint8_t* n_neighbors, const uint64_t* neighbors);
    void initialize_grid(
        uint8_t n_dims, const uint32_t* shape, uint8_t max_neighbors, int16_t** neighbor_offsets
    );
    void initialize_potentials(const double* potentials0, const double* potentials1);

public:
    // Rectangle-shaped grid. This constructor constructs the adjacency list
    // internally.
    BinaryBeliefPropagation(
        uint8_t n_dims,
        const uint32_t* shape,
        uint8_t max_neighbors,
        int16_t** neighbor_offsets,
        const double* potentials0,
        const double* potentials1,
        double p,
        double q
    );
    // Constructor that allows arbitrary graph structures (not necessarily a grid).
    // Note that the `neighbors` parameter is a 1D list because it is difficult
    // to pass a jagged array with Cython.
    BinaryBeliefPropagation(
        uint64_t n_nodes,
        const uint8_t* n_neighbors,
        const uint64_t* neighbors,
        const double* potentials0,
        const double* potentials1,
        double p,
        double q
    );
    ~BinaryBeliefPropagation();

    void run(
        double precision, /*=.1*/
        uint16_t max_iter, /*=100*/
        double log_bound, /*=50.*/
        bool taylor_approximation, /*=false*/
        uint64_t n_threads /*=1*/
    );
    void marginals(double* res, double log_bound);
};

#endif  // BINARY_BELIEF_PROPAGATION_HPP_
