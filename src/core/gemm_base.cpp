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
void pack_matrix_A_col(int64_t m, int64_t k, TA *A, int64_t lda, int64_t offset, TA *packA) {
    int64_t i, p;
    TA *a_ptr[RM];

    for (i = 0; i < RM; ++i) {
        if (i < m) {
            a_ptr[i] = &A[offset + i];
        }
        else {
            a_ptr[i] = &A[offset + 0];
        }
    }

    for (p = 0; p < k; p++) {
        for (i = 0; i < RM; i++) {
            *packA = *a_ptr[i];
            packA++;
            a_ptr[i] += lda;
        }
    }
};

template <typename TB, int64_t RM, int64_t RN>
void pack_matrix_B(int64_t k, int64_t n, TB *B, int64_t ldb, int64_t offset, TB *packB) {
    int64_t j, p;
    TB *b_ptr[RN];

    for (j = 0; j < RN; ++j) {
        if (j < n) {
            b_ptr[j] = &B[ldb * (offset + j)];
        }
        else {
            b_ptr[j] = &B[ldb * (offset + 0)];
        }
    }

    for (p = 0; p < k; p++) {
        for (j = 0; j < RN; j++) {
            *packB = b_ptr[j][p];
            packB++;
        }
    }
}



}
}

