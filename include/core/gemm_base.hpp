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

template <typename T, int64_t MR, int64_t NR>
void pack_matrix_A(int64_t m, int64_t k, T *A, int64_t lda, int64_t offset, T *packA);

template <typename T, int64_t MR, int64_t NR>
void pack_matrix_B(int64_t k, int64_t n, T *B, int64_t ldb, int64_t offset, T *packB);

}
}
#endif // GEMM_BASE_HPP_