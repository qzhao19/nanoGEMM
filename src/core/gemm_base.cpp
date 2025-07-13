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
void pack_matrix_A(int64_t m, int64_t k, const TA *A, int64_t lda, int64_t offset, TA *packA) {
    int64_t i, p;
    const TA *a_ptr[RM];

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
void pack_matrix_B(int64_t k, int64_t n, const TB *B, int64_t ldb, int64_t offset, TB *packB) {
    int64_t j, p;
    const TB *b_ptr[RN];

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
};

template float* gemm::detail::malloc_aligned<float>(int64_t, int64_t, int64_t);
template double* gemm::detail::malloc_aligned<double>(int64_t, int64_t, int64_t);

template void gemm::detail::pack_matrix_A<float, 4, 4>(int64_t, int64_t, const float*, int64_t, int64_t, float*);
template void gemm::detail::pack_matrix_A<double, 4, 4>(int64_t, int64_t, const double*, int64_t, int64_t, double*);

template void gemm::detail::pack_matrix_B<float, 4, 4>(int64_t, int64_t, const float*, int64_t, int64_t, float*);
template void gemm::detail::pack_matrix_B<double, 4, 4>(int64_t, int64_t, const double*, int64_t, int64_t, double*);

}
}

