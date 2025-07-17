#ifndef GEMM_BASE_HPP_
#define GEMM_BASE_HPP_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>

namespace gemm {
namespace detail {

#define MEMORY_ALIGNMENT 32
#define UNROLLING_SIZE 16

template <typename T>
T* malloc_aligned(int64_t m, int64_t n, int64_t size) {
    int64_t bytes = m * n * size;
    if (bytes % MEMORY_ALIGNMENT != 0) {
        bytes += MEMORY_ALIGNMENT - (bytes % MEMORY_ALIGNMENT);
    }

    void *ptr = std::aligned_alloc(MEMORY_ALIGNMENT, bytes);
    if (!ptr) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, m * n * size);

    return static_cast<T*>(ptr);
}; 

}
}
#endif // GEMM_BASE_HPP_