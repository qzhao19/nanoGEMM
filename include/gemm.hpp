#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <algorithm>
#include <core/gemm_base.hpp>
#include <arch/gemm_kernels.hpp>
#include <arch/x86/gemm_4x4_kernel.hpp>

namespace gemm {
namespace detail {

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

    void pack_matrix_A(int64_t m, int64_t k, const TA *A, TA *packA, int64_t offset) {  
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
    };

    void pack_matrix_B(int64_t k, int64_t n, const TB *B, TB *packB, int64_t offset) {
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
    };

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
    
    void multiply(int64_t m, int64_t n, int64_t k) {
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
}; 

}
} 
#endif // GEMM_HPP_