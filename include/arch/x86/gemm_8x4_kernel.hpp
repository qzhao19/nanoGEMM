#ifndef X86_GEMM_8X4_KERNEL_HPP_
#define X86_GEMM_8X4_KERNEL_HPP_

#include <arch/gemm_kernels.hpp>

namespace gemm {
namespace detail {

template <int64_t MR = 8, int64_t NR = 4>
inline void AddDot_8x4_kernel(int64_t k, float *a, float *b, float *c, int64_t ldc);

template <int64_t MR = 8, int64_t NR = 4>
inline void AddDot_8x4_kernel(int64_t k, double *a, double *b, double *c, int64_t ldc);

}
}
#endif // X86_GEMM_8X4_KERNEL_HPP_