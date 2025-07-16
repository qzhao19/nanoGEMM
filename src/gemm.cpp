#include <gemm.hpp>

namespace gemm {

template <typename TA, typename TB, typename TC, 
          int64_t RM, int64_t RN,
          int64_t CM, int64_t CK, int64_t CN>
void GEMM<TA, TB, TC, RM, RN, CM, CK, CN>::pack_matrix_A(
    int64_t m, int64_t k, 
    const TA *A, 
    TA *packA, 
    int64_t offset) {
    
    int64_t i, p;
    const TA *a_ptr[RM];

    for (i = 0; i < RM; ++i) {
        if (i < m) {
            a_ptr[i] = &A[offset + i];
        }
        else {
            a_ptr[i] = &A[offset + 0];
        }
    }

    for (p = 0; p < k; p++) {
        for (i = 0; i < RM; i++) {
            *packA = *a_ptr[i];
            packA++;
            a_ptr[i] += lda_;
        }
    }
}

template <typename TA, typename TB, typename TC, 
          int64_t RM, int64_t RN,
          int64_t CM, int64_t CK, int64_t CN>
void GEMM<TA, TB, TC, RM, RN, CM, CK, CN>::pack_matrix_B(
    int64_t k, int64_t n, 
    const TB *B, 
    TB *packB, 
    int64_t offset) {
    
    int64_t j, p;
    const TB *b_ptr[RN];

    for (j = 0; j < RN; ++j) {
        if (j < n) {
            b_ptr[j] = &B[ldb_ * (offset + j)];
        }
        else {
            b_ptr[j] = &B[ldb_ * (offset + 0)];
        }
    }

    for (p = 0; p < k; p++) {
        for (j = 0; j < RN; j++) {
            *packB = b_ptr[j][p];
            packB++;
        }
    }
}

template <typename TA, typename TB, typename TC, 
          int64_t RM, int64_t RN,
          int64_t CM, int64_t CK, int64_t CN>
void GEMM<TA, TB, TC, RM, RN, CM, CK, CN>::multiply(int64_t m, int64_t n, int64_t k) {
    int64_t i, j, p;
    int64_t ic, jc, pc;
    int64_t min_m, min_k, min_n;

    TA *packA; 
    TB *packB;
    packA = gemm::detail::malloc_aligned<TA>(CK, CM + 1, sizeof(TA));
    packB = gemm::detail::malloc_aligned<TB>(CK, CN + 1, sizeof(TB));

    for (jc = 0; jc < n; jc += CN) {
        min_n = std::min(n - jc, CN);

        for (pc = 0; pc < k; pc += CK) {
            min_k = std::min(k - pc, CK);

            // packing sub-matrix B 
            for (j = 0; j < min_n; j += RN) {
                gemm::detail::pack_matrix_B(
                    min_k, 
                    std::min(min_n - j, RN),
                    &B_[pc],
                    &packB[j * min_k],
                    jc + j
                );
            }
            
            for (ic = 0; ic < m; ic += CM) {
                min_m = std::min(m - ic, CM);
                
                // packing sub-matrix A
                for (i = 0; i < min_m; i+= RM) {
                    gemm::detail::pack_matrix_A(
                        std::min(min_m - i, RM), 
                        min_k, 
                        &A_[pc * lda_], 
                        &packA[0 * CM * min_k + i * min_k],
                        ic + i
                    );
                }

                for (j = 0; j < n; j += RN) {
                    for (i = 0; i < m; i += RM) {
                        micro_kernel_(
                            k,
                            &packA[i * k],
                            &packB[j * k],
                            &C_[j * ldc_ + i],
                            ldc_
                        );
                    }
                }
            }
        }
    }
};

template class gemm::GEMM<float, float, float, 4, 4, 72, 256, 2048>;
template class gemm::GEMM<double, double, double, 4, 4, 72, 256, 2048>;

} // gemm
