#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <algorithm>
#include <string>
#include <core/gemm_base.hpp>
#include <core/kernel_factory.hpp>
#include <arch/gemm_kernels.hpp>
#include <arch/x86/gemm_4x4_kernel.hpp>
#include <arch/x86/gemm_8x4_kernel.hpp>

namespace gemm {
namespace detail {

template <typename TA, typename TB, typename TC, 
          int64_t RM, int64_t RN,
          int64_t CM, int64_t CK, int64_t CN>
class GEMM {
private:
    const TA *const A_;
    const TB *const B_;
    TC *const C_;

    const int64_t lda_;
    const int64_t ldb_;
    const int64_t ldc_;
    
    gemm::detail::MicroKernelType<TA, TB, TC, RM, RN> micro_kernel_;

    void pack_matrix_A(
        int64_t m, int64_t k, 
        int64_t row_offset, int64_t col_offset, 
        const TA *A, TA *packed_A) {  
        
        int64_t i, p;
        const TA *a_ptr[RM];

        for (i = 0; i < RM; ++i) {
            if (i < m) {
                a_ptr[i] = &A[(col_offset + 0) * lda_ + (row_offset + i)];
            }
            else {
                a_ptr[i] = &A[(col_offset + 0) * lda_ + (row_offset + 0)];
            }
        }

        for (p = 0; p < k; ++p) {
            // define a local var to store row elements
            TA a_val[RM];
            for (i = 0; i < RM; ++i) {
                if (i < m) {
                    // cache each row elements
                    // move ptr to next col 
                    a_val[i] = *a_ptr[i];
                    a_ptr[i] += lda_;
                }
                else {
                    a_val[i] = 0;
                }
            }
            
            // then write to the pack buffer all at once
            for (i = 0; i < RM; ++i) {
                *packed_A++ = a_val[i];
            }
        }
    };

    void pack_matrix_B(
        int64_t k, int64_t n, 
        int64_t row_offset, int64_t col_offset, 
        const TB *B, TB *packed_B) {
        
        int64_t j, p;
        const TB *b_ptr[RN];

        for (j = 0; j < RN; ++j) {
            if (j < n) {
                b_ptr[j] = &B[(col_offset + j) * ldb_ + row_offset];
            }
            else {
                b_ptr[j] = &B[(col_offset + 0) * ldb_ + row_offset];
            }
        }

        for (p = 0; p < k; p++) {
            // read all values into a local array (in the cache)
            TB b_val[RN];
            for (j = 0; j < RN; j++) {
                if (j < n) {  
                    // only access valid elements
                    b_val[j] = *b_ptr[j];
                    b_ptr[j]++;
                }
                else {  
                    // add proper zero padding
                    b_val[j] = 0;
                    // don't increment pointer for padding
                }
            }
            // write to the pack buffer all at once
            for (j = 0; j < RN; j++) {
                *packed_B++ = b_val[j];
            }
        }
    };

public:
    GEMM(const TA *A, int64_t lda, 
         const TB *B, int64_t ldb, 
         TC *C, int64_t ldc,
         gemm::detail::MicroKernelType<TA, TB, TC, RM, RN> micro_kernel) :
            A_(A), lda_(lda), 
            B_(B), ldb_(ldb), 
            C_(C), ldc_(ldc),
            micro_kernel_(micro_kernel) {};
    ~GEMM() = default;
    
    void multiply(int64_t m, int64_t n, int64_t k) {
        int64_t i, j, p;
        int64_t ic, jc, pc;
        int64_t ib, jb, pb;

        TA *packA; 
        TB *packB;
        packA = gemm::detail::malloc_aligned<TA>(CK, CM + 1, sizeof(TA));
        packB = gemm::detail::malloc_aligned<TB>(CK, CN + 1, sizeof(TB));

        for (jc = 0; jc < n; jc += CN) {
            jb = std::min(n - jc, CN);

            for (pc = 0; pc < k; pc += CK) {
                pb = std::min(k - pc, CK);

                // packing sub-matrix B 
                for (j = 0; j < jb; j += RN) {
                    pack_matrix_B(
                        pb,                      // number of rows to actually pack
                        std::min(jb - j, RN),    // number of columns to actually pack
                        pc,                      // global row offset
                        jc + j,                  // global columns offset
                        B_,                      // original matrix pointer
                        &packB[j * pb]           // packed buffer
                    );
                }
                
                for (ic = 0; ic < m; ic += CM) {
                    ib = std::min(m - ic, CM);
                    // packing sub-matrix A
                    for (i = 0; i < ib; i+= RM) {
                        pack_matrix_A(
                            std::min(ib - i, RM),   // number of rows to actually pack
                            pb,                     // umber of columns to actually pack
                            ic + i,                 // global row offset
                            pc,                     // global columns offset
                            A_,                     // original matrix pointer
                            &packA[i * pb]          // packed buffer position
                        );
                    }

                    for (j = 0; j < jb; j += RN) {
                        for (i = 0; i < ib; i += RM) {
                            micro_kernel_(
                                pb,
                                &packA[i * pb],
                                &packB[j * pb],
                                &C_[(jc + j) * ldc_ + (ic + i)],
                                ldc_
                            );
                        }
                    }
                }
            }
        }
        free(packA);
        free(packB);
    };
}; 

}

void matmul(int64_t m, int64_t n, int64_t k,
            const float *A, int64_t lda,
            const float *B, int64_t ldb,
            float *C, int64_t ldc, 
            const std::string &kernel);

void matmul(int64_t m, int64_t n, int64_t k,
            const double *A, int64_t lda,
            const double *B, int64_t ldb,
            double *C, int64_t ldc,
            const std::string &kernel);

} 
#endif // GEMM_HPP_