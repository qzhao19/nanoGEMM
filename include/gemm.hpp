#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <core/gemm_base.hpp>
#include <arch/gemm_kernels.hpp>

namespace gemm {

template <typename TA, typename TB, typename TC, 
          int64_t CM, int64_t CK, int64_t CN,
          int64_t RM, int64_t RN>
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
            C_(C), ldc_(ldc) {};
    ~GEMM() = default;
    
    void multiply(int64_t m, int64_t n, int64_t k);
}; 

} 
#endif // GEMM_HPP_