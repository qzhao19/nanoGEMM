#ifndef X86_GEMM_4X4_KERNEL_HPP_
#define X86_GEMM_4X4_KERNEL_HPP_

#include <arch/gemm_kernels.hpp>

namespace gemm {
namespace detail {

template <int64_t RM = 4, int64_t RN = 4>
inline void AddDot_4x4_kernel_float(int64_t k, float *a, float *b, float *c, int64_t ldc);

template <int64_t RM = 4, int64_t RN = 4>
inline void AddDot_4x4_kernel_double(int64_t k, double *a, double *b, double *c, int64_t ldc);

template <typename TA, typename TB, typename TC, int64_t RM = 4, int64_t RN = 4>
inline void AddDot_4x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc);

}
}
#endif // X86_GEMM_4X4_KERNEL_HPP_