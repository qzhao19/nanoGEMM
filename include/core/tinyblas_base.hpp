#ifndef TINYBLAS_BASE_HPP_
#define TINYBLAS_BASE_HPP_

#include <cmath>
#include <cassert>
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

template <typename T>
inline void free_aligned(T *ptr) {
    if (!ptr) return;
    std::free(ptr);
}


}  // namespace detail
}  // namespace tinyBLAS

#endif  // TINYBLAS_BASE_HPP_