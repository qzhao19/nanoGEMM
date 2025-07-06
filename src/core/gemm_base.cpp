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

template <typename TA, int64_t RM, int64_t RN>
void pack_matrix_A(int64_t m, int64_t k, TA *A, int64_t lda, int64_t offset, TA *packA) {
    int64_t i, p;
    TA *a_ptr[RM];

    for (i = 0; i < RM; ++i) {
        if (i < m) {
            a_ptr[i] = &A[offset + i];
        }
        else {
            a_ptr[i] = nullptr;
        }
    }

    for (p = 0; p < k; p++) {
        for (i = 0; i < RM; i++) {
            if (a_ptr[i]) {
                *packA = *a_ptr[i];
                a_ptr[i] = a_ptr[i] + lda;
            }
            else {
                *packA = 0;
            }
            packA++;
        }
    }
};

}
}

