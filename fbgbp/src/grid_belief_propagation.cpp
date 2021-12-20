#include "grid_belief_propagation.hpp"
#include "parallel_for.hpp"

GridBeliefPropagation::GridBeliefPropagation(
    uint8_t n_dims,
    const uint32_t* shape,
    const double* potentials0,
    const double* potentials1,
    double p,
    double q
) : n_dims(n_dims), p(p), q(q) {
    this->alpha = p / q;
    this->log_alpha = std::log(p) - std::log(q);
    this->n_nodes = 1;
    for (uint8_t i = 0; i < n_dims; i++)
        this->n_nodes *= shape[i];

    this->shape = new uint32_t[n_dims];
    std::copy(&shape[0], &shape[n_dims], this->shape);
    this->initialize_offsets();
    this->initialize_graph();
    this->initialize_potentials(potentials0, potentials1);
}

GridBeliefPropagation::~GridBeliefPropagation() {
    delete this->shape;
    delete this->offsets;
    delete this->n_neighbors;
    for (uint64_t i = 0; i < this->n_nodes; i++)
        delete[] this->neighbors[i];
    delete[] this->neighbors;
    delete[] this->message_index;
    delete[] this->messages;
    delete[] this->lambda;
}

void GridBeliefPropagation::initialize_offsets() {
    this->offsets = new uint32_t[this->n_dims];
    this->offsets[0] = 1;
    for (int i = 1; i < this->n_dims; i++) {
        uint32_t offset = 1;
        for (int j = - i; j < 0; j++)
            offset *= this->shape[this->n_dims + j];
        this->offsets[i] = offset;
    }
}

void GridBeliefPropagation::initialize_graph() {
    // Create n_nodes x n_dims array of node coordinates
    uint32_t* prod = new uint32_t[this->n_dims-1];
    for (uint8_t i = 0; i < this->n_dims - 1; i++) {
        prod[i] = 1;
        for (uint8_t j = i+1; j < this->n_dims; j++)
            prod[i] *= this->shape[j];
    }

    uint32_t** node_coordinates = new uint32_t*[this->n_nodes];
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        node_coordinates[i] = new uint32_t[this->n_dims];
        uint64_t remaining = i;
        for (int j = 0; j < this->n_dims - 1; j++) {
            node_coordinates[i][j] = remaining / prod[j];
            remaining %= prod[j];  // Does the compiler reuse previous result?

        }
        node_coordinates[i][this->n_dims-1] = remaining;
    }

    // Compute neighbors
    this->n_neighbors = new uint8_t[this->n_nodes];
    this->neighbors = new uint64_t*[this->n_nodes];
    uint64_t* neighbors_temp = new uint64_t[this->n_dims*2];
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        uint32_t* node_coords = node_coordinates[i];
        uint8_t n_neighbors = 0;
        for (int j = 0; j < this->n_dims; j++) {
            uint32_t i_temp = i - offsets[j];
            if (i_temp >= 0 && i_temp < this->n_nodes) {
                uint32_t* i_coords = node_coordinates[i_temp];
                uint8_t count0 = 0;
                uint8_t count1 = 0;
                for (int k = 0; k < this->n_dims; k++) {
                    count0 += node_coords[k] == i_coords[k];
                    count1 += (
                        node_coords[k] > i_coords[k] ?
                        (node_coords[k] - i_coords[k]) :
                        (i_coords[k] - node_coords[k])
                    ) == 1;
                }
                if (count0 == this->n_dims - 1 && count1 == 1) {
                    neighbors_temp[n_neighbors] = i_temp;
                    n_neighbors++;
                }
            }

            i_temp = i + offsets[j];
            if (i_temp >= 0 && i_temp < this->n_nodes) {
                uint32_t* i_coords = node_coordinates[i_temp];
                uint8_t count0 = 0;
                uint8_t count1 = 0;
                for (int k = 0; k < this->n_dims; k++) {
                    count0 += node_coords[k] == i_coords[k];
                    count1 += (
                        node_coords[k] > i_coords[k] ?
                        (node_coords[k] - i_coords[k]) :
                        (i_coords[k] - node_coords[k])
                    ) == 1;
                }
                if (count0 == this->n_dims - 1 && count1 == 1) {
                    neighbors_temp[n_neighbors] = i_temp;
                    n_neighbors++;
                }
            }
        }

        this->n_neighbors[i] = n_neighbors;
        this->neighbors[i] = new uint64_t[n_neighbors];
        std::copy(
            &neighbors_temp[0],
            &neighbors_temp[n_neighbors],
            &this->neighbors[i][0]
        );
    }

    delete prod;
    delete[] neighbors_temp;
    for (uint64_t i = 0; i < this->n_nodes; i++)
        delete[] node_coordinates[i];
    delete[] node_coordinates;
}

void GridBeliefPropagation::initialize_potentials(
    const double* potentials0, const double* potentials1
) {
    uint64_t n_messages = 0;
    double min_log = std::log(1e-6);
    this->lambda = new double[this->n_nodes];
    this->message_index = new uint64_t[this->n_nodes];
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        this->message_index[i] = n_messages;
        n_messages += this->n_neighbors[i];
        this->lambda[i] = (
            potentials1[i] < 1e-6 ? min_log : std::log(potentials1[i])
        ) - (potentials0[i] < 1e-6 ? min_log : std::log(potentials0[i]));
    }

    this->n_messages = n_messages;
    this->messages = new double[n_messages];
    for (uint64_t i = 0; i < n_messages; i++)
        this->messages[i] = 0.;
}

void GridBeliefPropagation::run(
    double precision=.1,
    uint16_t max_iter=100,
    double log_bound=100.,
    bool taylor_approximation=false,
    uint64_t n_threads=1
) {
    // messages will alternate to save space
    double* messages2 = new double[this->n_messages];
    double* last_messages = this->messages;
    double* next_messages = messages2;

    bool precision_reached = false;
    for (uint16_t n = 0; n < max_iter && !precision_reached; n++) {
        // Compute new messages
        parallel_for(this->n_nodes, n_threads, [&](uint64_t start, uint64_t end) {
            for (uint64_t to = start; to < end; to++) {
                uint64_t* to_neighbors = this->neighbors[to];
                for (uint8_t from_i = 0; from_i < this->n_neighbors[to]; from_i++) {
                    uint64_t from = to_neighbors[from_i];
                    uint64_t* from_neighbors = this->neighbors[from];
                    double message = this->lambda[from];
                    for (
                        uint8_t from_neighbor_i = 0;
                        from_neighbor_i < this->n_neighbors[from];
                        from_neighbor_i++
                    )
                        message += (
                            (double) (from_neighbors[from_neighbor_i] != to)
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
                        double c = std::exp(message);
                        message = std::log((this->q + this->p * c) / (this->p + this->q * c));
                    }
                    next_messages[from_i + this->message_index[to]] = message;
                }
            }
        });
        std::swap(last_messages, next_messages);

        double diff = 0;
        for (uint64_t i = 0; i < n_messages; i++)
            diff += std::pow(last_messages[i] - next_messages[i], 2);
        diff /= (double) n_messages;
        precision_reached = diff < precision;
    }
    this->messages = last_messages;
    delete[] next_messages;
}

void GridBeliefPropagation::marginals(double* res) {
    for (uint64_t i = 0; i < this->n_nodes; i++) {
        double denom = -this->lambda[i];
        uint64_t* neighbors = this->neighbors[i];
        for (uint8_t j = 0; j < this->n_neighbors[i]; j++)
            denom -= this->messages[this->message_index[i] + j];
        res[i] = 1. / (1. + std::exp(denom));
    }
}
