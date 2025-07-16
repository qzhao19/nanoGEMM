#include <core/gemm_base.hpp>

namespace gemm {
namespace detail {

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

template float* gemm::detail::malloc_aligned<float>(int64_t, int64_t, int64_t);
template double* gemm::detail::malloc_aligned<double>(int64_t, int64_t, int64_t);

}
}

