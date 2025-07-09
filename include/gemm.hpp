#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <core/gemm_base.hpp>

namespace gemm {

template <typename TA, typename TB, typename TC, 
          int64_t RM = 4, int64_t RN = 1, 
          int64_t CM = 72, int64_t CK = 256, int64_t CN = 1020>
class GEMM {
private:
    using MicroKernelType = std::function<void(int, TA*, TB*, TC*, int)>;
    const TA *const A_;
    const TB *const B_;
    TC *const C_;

    const int64_t lda_;
    const int64_t ldb_;
    const int64_t ldc_;
    MicroKernelType<TA, TB, TC, RM, RN> micro_kernel_;

public:
    GEMM(const TA *A, int64_t lda, 
         const TB *B, int64_t ldb, 
         TC *C, int64_t ldc, 
         MicroKernelType<TA, TB, TC, RM, RN> micro_kernel) :
            A_(A), lda_(lda), 
            B_(B), ldb_(ldb), 
            C_(C), ldc_(ldc), 
            micro_kernel_(micro_kernel) {};
    ~GEMM() = default;
    
    void multiply(int64_t m, int64_t n, int64_t k);
}; 

} 
#endif // GEMM_HPP_