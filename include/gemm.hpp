#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <algorithm>
#include <core/gemm_base.hpp>
#include <arch/gemm_kernels.hpp>
#include <arch/x86/gemm_4x4_kernel.hpp>

namespace gemm {

template <typename TA, typename TB, typename TC, 
          int64_t RM, int64_t RN,
          int64_t CM = 72, int64_t CK = 256, int64_t CN = 2048>
class GEMM {
private:
    const TA *const A_;
    const TB *const B_;
    TC *const C_;

    const int64_t lda_;
    const int64_t ldb_;
    const int64_t ldc_;
    gemm::detail::MicroKernelType<TA, TB, TC, RM, RN> micro_kernel_;

public:
    GEMM(const TA *A, int64_t lda, 
         const TB *B, int64_t ldb, 
         TC *C, int64_t ldc) :
            A_(A), lda_(lda), 
            B_(B), ldb_(ldb), 
            C_(C), ldc_(ldc) {
        micro_kernel_ = gemm::detail::AddDot_4x4_kernel<TA, TB, TC>;
    };
    ~GEMM() = default;
    
    void multiply(int64_t m, int64_t n, int64_t k);
}; 

} 
#endif // GEMM_HPP_