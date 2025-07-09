#include <gemm.hpp>
#include <core/gemm_base.hpp>
#include <arch/gemm_kernels.hpp>

namespace gemm {

template<typename TA, typename TB, typename TC, 
          int64_t RM, int64_t RN, 
          int64_t CM, int64_t CK, int64_t CN>    
void GEMM<TA, TB, TC, RM, RN, CM, CK, CN>::multiply(int64_t m, int64_t n, int64_t k) {
    int64_t i, j, p;
    int64_t ic, jc, pc;
    int64_t ib, jb, pb;
    int64_t min_m, min_k, min_n;

    TA *packA; 
    TB *packB;
    packA = malloc_aligned<TA>(CK, CM + 1, sizeof(TA));
    packB = malloc_aligned<TB>(CK, CN + 1, sizeof(TB));

    for (jc = 0; jc < n; jc += CN) {
        jb = std::min(n - jc, CN);

        for (pc = 0; pc < k; pc += CK) {
            pb = std::min(k - pc, CK);

            // packing sub-matrix B 
            for (j = 0; j < jb; j += RN) {
                gemm::detail::pack_matrix_B<TB, RM, RN>(
                    std::min(jb - j, RN),
                    pb, 
                    &B_[pc],
                    k,
                    jc + j,
                    &packB[j * pb]
                );
            }
            


        }


    }


};

}