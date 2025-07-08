#include <gemm.hpp>
#include <core/gemm_base.hpp>
#include <arch/gemm_kernels.hpp>

namespace gemm {

template<typename TA, typename TB, typename TC, 
          int RM, int RN, 
          int CM, int CK, int CN>    
void GEMM<TA, TB, TC, RM, RN, CM, CK, CN>::multiply(int m, int n, int k) {
    int i, j, p;
    int ic, jc, pc;
    int min_m, min_k, min_n;

    TA *packA; 
    TB *packB;




};

}