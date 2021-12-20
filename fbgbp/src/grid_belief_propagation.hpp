#ifndef GRID_BELIEF_PROPAGATION_HPP_
#define GRID_BELIEF_PROPAGATION_HPP_
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

class GridBeliefPropagation {
private:
    uint64_t n_threads;
    uint64_t n_nodes;
    uint8_t n_dims;
    uint32_t* shape;

    // Variables defining edge/neighborhood structure.
    uint32_t* offsets;
    uint8_t* n_neighbors;
    uint64_t** neighbors;

    // Params
    double p;  // xi == xj
    double q;  // xi != xj
    double alpha; // p / q
    double log_alpha;  // log(p / q)

    // Messages -- messages are indexed in the order they appear in the
    // neighbors array. For example, for node i if we have neighbors[i][0] == j,
    // and this is the k'th element in the array, then messages[k] is the message
    // sent j -> i.
    uint64_t n_messages;
    double* messages;
    uint64_t* message_index;
    double* lambda;

    void initialize_offsets();
    void initialize_graph();
    void initialize_potentials(const double* potentials0, const double* potentials1);

public:
    GridBeliefPropagation(
        uint8_t n_dims,
        const uint32_t* shape,
        const double* potentials0,
        const double* potentials1,
        double p,
        double q
    );
    ~GridBeliefPropagation();

    void run(
        double precision, /*=.1*/
        uint16_t max_iter, /*=100*/
        double log_bound, /*=100.*/
        bool taylor_approximation, /*=false*/
        uint64_t n_threads /*=1*/
    );
    void marginals(double* res);
};

#endif  // GRID_BELIEF_PROPAGATION_HPP_
