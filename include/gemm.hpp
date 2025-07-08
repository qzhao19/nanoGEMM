#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <core/gemm_base.hpp>

namespace gemm {

template <typename TA, typename TB, typename TC, 
          int RM = 4, int RN = 1, 
          int CM = 72, int CK = 256, int CN = 1020>
class GEMM {
private:
    const TA *const A_;
    const TB *const B_;
    TC *const C_;

    const int lda_;
    const int ldb_;
    const int ldc_;
    MicroKernelType<TA, TB, TC, RM, RN> micro_kernel_;

public:
    GEMM(const TA *A, int lda, 
         const TB *B, int ldb, 
         TC *C, int ldc, 
         MicroKernelType<TA, TB, TC, RM, RN> micro_kernel) :
            A_(A), lda_(lda), 
            B_(B), ldb_(ldb), 
            C_(C), ldc_(ldc), 
            micro_kernel_(micro_kernel) {};
    ~GEMM() = default;
    
    void multiply(int m, int n, int k);
}; 

} 
#endif // GEMM_HPP_