#ifndef TINYBLAS_GEMM_HPP_
#define TINYBLAS_GEMM_HPP_

#include <algorithm>
#include <arch/tinyblas_kernels.hpp>
#include <arch/x86/tinyblas_gemm_4x4_kernel.hpp>
#include <arch/x86/tinyblas_gemm_8x4_kernel.hpp>
#include <core/kernel_factory.hpp>
#include <core/tinyblas_base.hpp>
#include <string>

namespace tinyBLAS {
namespace detail {

template <typename TA,
          typename TB,
          typename TC,
          int64_t RM,
          int64_t RN,
          int64_t CM,
          int64_t CK,
          int64_t CN>
class GEMM {
private:
    const TA *const A_;
    const TB *const B_;
    TC *const C_;

    const int64_t lda_;
    const int64_t ldb_;
    const int64_t ldc_;

    MicroKernelType<TA, TB, TC, RM, RN> micro_kernel_;

    void pack_matrix_A(
        int64_t m, int64_t k, int64_t row_begin, int64_t col_begin, const TA *A, TA *packed_A) {
        int64_t i, p;
        // initialize the pointer array to store start ptr of each row
        const TA *a_ptr[RM];
        for (i = 0; i < std::min(m, RM); ++i) {
            a_ptr[i] = &A[(row_begin + i) * lda_ + col_begin];
        }

        for (p = 0; p < k; ++p) {
            // copy elements from sub-matrix A to packed_A
            for (i = 0; i < m; ++i) {
                *packed_A++ = a_ptr[i][p];
            }
            // pad with zeros to ensure each colmun has exactly RN elements
            for (; i < RM; ++i) {
                *packed_A++ = static_cast<TA>(0);
            }
        }
    };

    
    void pack_matrix_B(
        int64_t k, int64_t n, int64_t row_begin, int64_t col_begin, const TB *B, TB *packed_B) {
        int64_t p, j;
        const TB *b_ptr[k];
        
        // set the correct starting pointer for each row
        for (p = 0; p < k; ++p) {
            b_ptr[p] = &B[(row_begin + p) * ldb_ + col_begin];
        }

        for (p = 0; p < k; ++p) {
            const TB *row = b_ptr[p];
            for (j = 0; j < n; ++j) {
                *packed_B++ = row[j]; 
            }
            // pad with zeros to ensure each row has exactly RN elements
            for (; j < RN; ++j) {
                *packed_B++ = static_cast<TB>(0);
            }
        }
    };

public:
    GEMM(const TA *A,
         int64_t lda,
         const TB *B,
         int64_t ldb,
         TC *C,
         int64_t ldc,
         MicroKernelType<TA, TB, TC, RM, RN> micro_kernel)
        : A_(A), lda_(lda), B_(B), ldb_(ldb), C_(C), ldc_(ldc), micro_kernel_(micro_kernel){};
    ~GEMM() = default;

    void multiply(int64_t m, int64_t n, int64_t k) {
        // number of threads of inner dimension
        int64_t ic_nts;
        int64_t i, j, p;
        int64_t ic, jc, pc;
        int64_t ib, jb, pb;
        char *str;

        // check if the environment variable exists
        ic_nts = 1;
        str = std::getenv("IC_NTS");
        if (str != nullptr) {
            ic_nts = std::strtol(str, nullptr, 10);
        }

        TA *packed_A = malloc_aligned<TA>(CK, (CM + 1) * ic_nts, sizeof(TA));
        TB *packed_B = malloc_aligned<TB>(CK, (CN + 1) * ic_nts, sizeof(TB));

        // 5-th loop around micro-kernel
        for (ic = 0; ic < m; ic += CM) {
            ib = std::min(m - ic, CM);

            // 4-th loop around micro-kernel
            for (pc = 0; pc < k; pc += CK) {
                pb = std::min(k - pc, CK);

                // packing sub-matrix B
                #pragma omp parallel for num_threads(ic_nts)
                for (i = 0; i < ib; i += RM) {
                    pack_matrix_A(
                        std::min(ib - i, RM),
                        pb,
                        ic + i,
                        pc,
                        A_,
                        &packed_A[i * pb]
                    );
                }

                #pragma omp parallel num_threads(ic_nts) private(jc, jb, j) 
                {
                    int64_t thread_id = omp_get_thread_num();
                    // call partition_workload_by_thread
                    int64_t range_start, range_end;
                    partition_workload_by_thread(n, RN, range_start, range_end);
                    // 3-rd loop around micro-kernel
                    for (jc = range_start; jc < range_end; jc += CN) {
                        jb = std::min(range_end - jc, CN);
                        // packing sub-matrix B
                        for (j = 0; j < jb; j += RN) {
                            pack_matrix_B(
                                pb,
                                std::min(jb - j, RN),
                                pc,
                                jc + j,
                                B_,
                                &packed_B[thread_id * CN * pb + j * pb]
                            );
                        }
                        // define micro-kernel ctx
                        MicroKernelCtxType<TB> ctx;
                        ctx.next = packed_B;
                        // 2-th loop around micro-kernel
                        for (i = 0; i < ib; i += RM) {
                            ctx.m = std::min(ib - i, RM);
                            if (i + RM > ib) {
                                ctx.next += pb * RN;
                            }
                            for (j = 0; j < jb; j += RN) {
                                ctx.n = std::min(jb - j, RN);
                                micro_kernel_(
                                    pb,
                                    &packed_A[i * pb],
                                    &packed_B[thread_id * CN * pb + j * pb],
                                    &C_[(ic + i) * ldc_ + (jc + j)],
                                    ldc_,
                                    &ctx
                                );

                            }
                        }
                    }
                }
            }
        }
        free(packed_A);
        free(packed_B);
    };
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // TINYBLAS_GEMM_HPP_