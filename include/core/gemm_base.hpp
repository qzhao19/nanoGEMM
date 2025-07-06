#ifndef GEMM_BASE_HPP_
#define GEMM_BASE_HPP_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

namespace gemm {
namespace detail {

#define MEMORY_ALIGNMENT 32
#define UNROLLING_SIZE 16

template <typename T>
T* malloc_aligned(int64_t m, int64_t n, int64_t size);

template <typename TA, int64_t RM, int64_t RN>
void pack_matrix_A(int64_t m, int64_t k, TA *A, int64_t lda, int64_t offset, TA *packA);

template <typename TB, int64_t RM, int64_t RN>
void pack_matrix_B(int64_t k, int64_t n, TB *B, int64_t ldb, int64_t offset, TB *packB);

}
}
#endif // GEMM_BASE_HPP_