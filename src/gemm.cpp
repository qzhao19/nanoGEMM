#include <gemm.hpp>
#include <core/gemm_base.hpp>
#include <arch/gemm_kernels.hpp>

namespace gemm {

template<typename TA, typename TB, typename TC, 
         int64_t CM, int64_t CK, int64_t CN,
         int64_t RM, int64_t RN>    
void GEMM<TA, TB, TC, CM, CK, CN, RM, RN>::multiply(int64_t m, int64_t n, int64_t k) {
    int64_t i, j, p;
    int64_t ic, jc, pc;
    int64_t min_m, min_k, min_n;

    TA *packA; 
    TB *packB;
    packA = malloc_aligned<TA>(CK, CM + 1, sizeof(TA));
    packB = malloc_aligned<TB>(CK, CN + 1, sizeof(TB));

    for (jc = 0; jc < n; jc += CN) {
        min_n = std::min(n - jc, CN);

        for (pc = 0; pc < k; pc += CK) {
            min_k = std::min(k - pc, CK);

            // packing sub-matrix B 
            for (j = 0; j < min_n; j += RN) {
                gemm::detail::pack_matrix_B<TB, RM, RN>(
                    min_k, 
                    std::min(min_n - j, RN),
                    &B_[pc],
                    k,
                    jc + j,
                    &packB[j * min_k]
                );
            }
            
            for (ic = 0; ic < m; ic += CM) {
                min_m = std::min(m - ic, CM);
                
                // packing sub-matrix A
                for (i = 0; i < min_m; i+= RM) {
                    gemm::detail::pack_matrix_A<TB, RM, RN>(
                        min(min_m - i, RM), 
                        min_k, 
                        &A[pc * lda], 
                        m, 
                        ic + i, 
                        &packA[0 * CM * min_k + i * min_k]
                    );
                }

                for (j = 0; j < n; j += RN) {
                    for (i = 0; i < m; i += RM) {
                        micro_kernel_(
                            k,
                            &packA[i * k],
                            &packB[j * k],
                            &C[j * ldc_ + i],
                            ldc
                        )
                    }
                }
            }
        }
    }
};

}