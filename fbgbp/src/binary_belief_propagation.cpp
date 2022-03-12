#include "binary_belief_propagation.hpp"
#include "parallel_for.hpp"

BinaryBeliefPropagation::BinaryBeliefPropagation(
    uint8_t n_dims,
    const uint32_t* shape,
    uint8_t max_neighbors,
    int16_t** neighbor_offsets,
    const double* potentials0,
    const double* potentials1,
    double p,
    double q
) {
    this->p = (float) p;
    this->q = (float) q;
    this->n_nodes = 1;
    for (uint8_t i = 0; i < n_dims; i++)
        this->n_nodes *= shape[i];

    this->initialize_alpha(p, q);
    this->initialize_grid(
        n_dims, shape, max_neighbors, neighbor_offsets
    );
    this->initialize_potentials(potentials0, potentials1);
}

BinaryBeliefPropagation::BinaryBeliefPropagation(
    uint64_t n_nodes,
    const uint8_t* n_neighbors,
    const uint64_t* neighbors,
    const double* potentials0,
    const double* potentials1,
    double p,
    double q
) : n_nodes(n_nodes) {
    this->p = (float) p;
    this->q = (float) q;
    this->initialize_alpha(p, q);
    this->initialize_neighbors(n_neighbors, neighbors);
    this->initialize_potentials(potentials0, potentials1);
}

BinaryBeliefPropagation::~BinaryBeliefPropagation() {
    delete[] this->n_neighbors;
    for (uint64_t i = 0; i < this->n_nodes; i++)
        delete[] this->neighbors[i];
    delete[] this->neighbors;
    delete[] this->message_index;
    delete[] this->messages;
    delete[] this->lambda;
}

void BinaryBeliefPropagation::initialize_alpha(double p, double q) {
    this->alpha = (float) (p / q);
    this->log_alpha = (float) (std::log(p) - std::log(q));
}

void BinaryBeliefPropagation::initialize_neighbors(
    const uint8_t* n_neighbors, const uint64_t* neighbors
) {
    this->n_neighbors = new uint8_t[this->n_nodes];
    std::copy(&n_neighbors[0], &n_neighbors[this->n_nodes], &this->n_neighbors[0]);

    this->neighbors = new uint64_t*[this->n_nodes];
    uint64_t neighbors_i = 0;
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        uint8_t _n_neighbors = this->n_neighbors[i];
        this->neighbors[i] = new uint64_t[_n_neighbors];

        std::copy(
            &neighbors[neighbors_i],
            &neighbors[neighbors_i + _n_neighbors],
            &this->neighbors[i][0]
        );
        neighbors_i += _n_neighbors;
    }
}

void BinaryBeliefPropagation::initialize_grid(
    uint8_t n_dims,
    const uint32_t* shape,
    uint8_t max_neighbors,
    int16_t** neighbor_offsets
) {
    // Create n_nodes x n_dims array of node coordinates
    uint32_t* prod = new uint32_t[n_dims];
    for (uint8_t i = 0; i < n_dims; i++) {
        prod[i] = 1;
        for (uint8_t j = i+1; j < n_dims; j++)
            prod[i] *= shape[j];
    }

    uint32_t** node_coordinates = new uint32_t*[this->n_nodes];
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        node_coordinates[i] = new uint32_t[n_dims];
        uint64_t remaining = i;
        for (uint8_t j = 0; j < n_dims; j++) {
            node_coordinates[i][j] = remaining / prod[j];
            remaining %= prod[j];  // Does the compiler reuse previous result?
        }
    }

    // Compute neighbors
    this->n_neighbors = new uint8_t[this->n_nodes];
    this->neighbors = new uint64_t*[this->n_nodes];

    uint64_t* neighbors_temp = new uint64_t[max_neighbors];
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        uint8_t n_neighbors = 0;
        uint32_t* node_coords = node_coordinates[i];
        for (uint8_t j = 0; j < max_neighbors; j++) {
            bool valid = true;
            int16_t* neighbor_offset = neighbor_offsets[j];
            uint32_t i_temp = 0;
            for (uint8_t k = 0; k < n_dims; k++) {
                int64_t coord = ((int64_t) node_coords[k]) + ((int64_t) neighbor_offset[k]);
                valid = valid && coord >= 0 && coord < (int64_t) shape[k];
                i_temp += coord * prod[k];
            }
            neighbors_temp[n_neighbors] = i_temp;
            n_neighbors += (uint8_t) valid;
        }

        this->n_neighbors[i] = n_neighbors;
        this->neighbors[i] = new uint64_t[n_neighbors];
        std::copy(
            &neighbors_temp[0],
            &neighbors_temp[n_neighbors],
            &this->neighbors[i][0]
        );
    }

    delete[] prod;
    delete[] neighbors_temp;
    for (uint64_t i = 0; i < this->n_nodes; i++)
        delete[] node_coordinates[i];
    delete[] node_coordinates;
}

void BinaryBeliefPropagation::initialize_potentials(
    const double* potentials0, const double* potentials1
) {
    uint64_t n_messages = 0;
    float min_log = std::log(1e-6);
    this->lambda = new float[this->n_nodes];
    this->message_index = new uint64_t[this->n_nodes];
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        this->message_index[i] = n_messages;
        n_messages += this->n_neighbors[i];
        this->lambda[i] = (float) ((
            potentials1[i] < 1e-6 ? min_log : std::log(potentials1[i])
        ) - (potentials0[i] < 1e-6 ? min_log : std::log(potentials0[i])));
    }

    this->n_messages = n_messages;
    this->messages = new float[n_messages];
    for (uint64_t i = 0; i < n_messages; i++)
        this->messages[i] = 0.;
}

void BinaryBeliefPropagation::run(
    double precision=.1,
    uint16_t max_iter=100,
    double log_bound=50.,
    bool taylor_approximation=false,
    uint64_t n_threads=1
) {
    // messages will alternate to save space
    float* messages2 = new float[this->n_messages];
    float* last_messages = this->messages;
    float* next_messages = messages2;

    bool precision_reached = false;
    for (uint16_t n = 0; n < max_iter && !precision_reached; n++) {
        // Compute new messages
        parallel_for(this->n_nodes, n_threads, [&](uint64_t start, uint64_t end) {
            for (uint64_t to = start; to < end; to++) {
                uint64_t* to_neighbors = this->neighbors[to];
                for (uint8_t from_i = 0; from_i < this->n_neighbors[to]; from_i++) {
                    uint64_t from = to_neighbors[from_i];
                    uint64_t* from_neighbors = this->neighbors[from];
                    float message = this->lambda[from];
                    for (
                        uint8_t from_neighbor_i = 0;
                        from_neighbor_i < this->n_neighbors[from];
                        from_neighbor_i++
                    )
                        message += (
                            (float) (from_neighbors[from_neighbor_i] != to)
                        ) * last_messages[from_neighbor_i + this->message_index[from]];

                    // If messages (which is actually the log-message) is
                    // below/above a certain threshold, clip to bound to
                    // prevent underflow/overflow.
                    if (message < -log_bound) {
                        message = -this->log_alpha;
                    } else if (message > log_bound) {
                        message = this->log_alpha;
                    } else if (taylor_approximation) {
                        // First two terms of Taylor series about x = 0.
                        message = message * (this->alpha - 1) / (this->alpha + 1)
                            - std::pow(message, 3) * (this->alpha * (this->alpha - 1))
                            / (3 * std::pow(1 + this->alpha, 3));
                    } else {
                        float c = std::exp(message);
                        message = std::log((this->q + this->p * c) / (this->p + this->q * c));
                    }
                    next_messages[from_i + this->message_index[to]] = message;
                }
            }
        });
        std::swap(last_messages, next_messages);

        float diff = 0;
        for (uint64_t i = 0; i < n_messages; i++)
            diff += std::pow(last_messages[i] - next_messages[i], 2);
        diff /= (float) n_messages;
        precision_reached = diff < precision;
    }
    this->messages = last_messages;
    delete[] next_messages;
}

void BinaryBeliefPropagation::marginals(double* res, double log_bound=50.) {
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        double denom = (double) -this->lambda[i];
        uint64_t* neighbors = this->neighbors[i];
        for (uint8_t j = 0; j < this->n_neighbors[i]; j++)
            // Denom is in log scale.
            denom -= (double) this->messages[this->message_index[i] + j];

        // Deal with under/overflow.
        if (denom < -log_bound) {
            res[i] = 1.;
        } else if (denom > log_bound) {
            res[i] = 0.;
        } else {
            res[i] = 1. / (1. + std::exp(denom));
        }
    }
}
