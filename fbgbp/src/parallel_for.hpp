#ifndef PARALLEL_FOR_HPP_
#define PARALLEL_FOR_HPP_
#include <algorithm>
#include <cstdint>
#include <thread>
#include <functional>
#include <vector>

/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
///
///
static
void parallel_for(uint64_t nb_elements, uint64_t nb_threads,
                  std::function<void (uint64_t start, uint64_t end)> functor)
{
    // -------
    bool use_threads = nb_threads > 1;

    uint64_t batch_size = nb_elements / nb_threads;
    uint64_t batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);

    if( use_threads )
    {
        // Multithread execution
        for(uint64_t i = 0; i < nb_threads; ++i)
        {
            uint64_t start = i * batch_size;
            my_threads[i] = std::thread(functor, start, start+batch_size);
        }
    }
    else
    {
        // Single thread execution (for easy debugging)
        for(uint64_t i = 0; i < nb_threads; ++i){
            uint64_t start = i * batch_size;
            functor( start, start+batch_size );
        }
    }

    // Deform the elements left
    uint64_t start = nb_threads * batch_size;
    functor( start, start+batch_remainder);

    // Wait for the other thread to finish their task
    if( use_threads )
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}
#endif  // PARALLEL_FOR_HPP_
