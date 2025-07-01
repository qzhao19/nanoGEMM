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
T* malloc_aligned(std::int64_t m, std::int64_t n, std::int64_t size);

template <typename T, std::int64_t MR, std::int64_t NR>
void pack_matrix_A(std::int64_t m, std::int64_t k, T *A, std::int64_t lda, std::int64_t offset, T *packA);

template <typename T, std::int64_t MR, std::int64_t NR>
void pack_matrix_B(std::int64_t k, std::int64_t n, T *B, std::int64_t ldb, std::int64_t offset, T *packB);

}
}
#endif // GEMM_BASE_HPP_