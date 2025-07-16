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
T* malloc_aligned(int64_t m, int64_t n, int64_t size);

}
}
#endif // GEMM_BASE_HPP_