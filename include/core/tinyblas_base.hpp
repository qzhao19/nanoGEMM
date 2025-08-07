#ifndef TINYBLAS_BASE_HPP_
#define TINYBLAS_BASE_HPP_

#include <omp.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <core/tinyblas_common.hpp>

namespace tinyBLAS {
namespace detail {

template <typename T>
inline T *malloc_aligned(int64_t m, int64_t n, int64_t size) {
    int64_t bytes = m * n * size;
    if (bytes % MEMORY_ALIGNMENT != 0) {
        bytes += MEMORY_ALIGNMENT - (bytes % MEMORY_ALIGNMENT);
    }

    void *ptr = std::aligned_alloc(MEMORY_ALIGNMENT, bytes);
    if (!ptr) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, m * n * size);

    return static_cast<T *>(ptr);
};

/**
 * ivide the total workload total_elems by the block_size,
 * and distribute it as evenly as possible among multiple
 * threads while maintaining block alignment.
 *
 * Allocation Strategy
 *  1. 2-level grouping: divide threads into low-index group and high-index group
 *  2. priority allocation: remainder blocks are prioritized to be allocated to low-index threads
 *  3. corner case handling: remaining work that is less than one block is allocated to the last
 * thread
 */
inline void partition_workload_by_thread(int64_t total_elems,
                                         int64_t block_size,
                                         int64_t &range_start,
                                         int64_t &range_end) {
    int64_t num_threads = omp_get_num_threads();
    int64_t thread_id = omp_get_thread_num();

    int64_t global_start = 0;
    int64_t global_end = total_elems;

    // actual workload that needs to be allocated before
    int64_t workload_size = global_end - global_start;

    // the total number of blocks
    // the remaining workload that is less than one block
    int64_t num_blocks_total = workload_size / block_size;
    int64_t num_leftover_elems = workload_size % block_size;

    // init the number of blocks allocated to individual thread, assuming an even distribution first
    int64_t num_blocks_per_hi_thread = num_blocks_total / num_threads;
    int64_t num_blocks_per_lo_thread = num_blocks_total / num_threads;

    // compute the remainder that cannot be evenly distributed,
    // i.e., the number of threads of low index that need to be allocated additional blocks
    int64_t num_lo_threads_extra_block = num_blocks_total % num_threads;

    // if there is a remainder,
    // ++1 to the number of blocks allocated to each thread in the low-index thread group
    if (num_lo_threads_extra_block != 0) {
        ++num_blocks_per_lo_thread;
    }

    // compute actual workload of individual thread
    // in the low thread group and high thread group
    int64_t workload_per_lo_thread = num_blocks_per_lo_thread * block_size;
    int64_t workload_per_hi_thread = num_blocks_per_hi_thread * block_size;

    // precompute the starting position of the low thread group
    // and the high thread group (after the low thread group)
    int64_t lo_group_start = global_start;
    int64_t hi_group_start = global_start + num_lo_threads_extra_block * workload_per_lo_thread;

    // if the current thread belongs to the low thread group, compute its working range
    if (thread_id < num_lo_threads_extra_block) {
        range_start = lo_group_start + (thread_id)*workload_per_lo_thread;
        range_end = lo_group_start + (thread_id + 1) * workload_per_lo_thread;
    } else {
        // if the current thread belongs to the high thread group, compute its working range
        range_start =
            hi_group_start + (thread_id - num_lo_threads_extra_block) * workload_per_hi_thread;
        range_end =
            hi_group_start + (thread_id - num_lo_threads_extra_block + 1) * workload_per_hi_thread;

        // if it is the last thread, allocate the remaining amount that is less than one block to it
        // as well
        if (thread_id == num_threads - 1) {
            range_end += num_leftover_elems;
        }
    }
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // TINYBLAS_BASE_HPP_