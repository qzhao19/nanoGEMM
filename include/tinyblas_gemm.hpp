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
        int64_t m, int64_t k, int64_t row_offset, int64_t col_offset, const TA *A, TA *packed_A) {
        int64_t i, p;
        const TA *a_ptr[RM];

        for (i = 0; i < RM; ++i) {
            if (i < m) {
                a_ptr[i] = &A[(col_offset + 0) * lda_ + (row_offset + i)];
            } else {
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
                } else {
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
        int64_t k, int64_t n, int64_t row_offset, int64_t col_offset, const TB *B, TB *packed_B) {
        int64_t j, p;
        const TB *b_ptr[RN];

        for (j = 0; j < RN; ++j) {
            if (j < n) {
                b_ptr[j] = &B[(col_offset + j) * ldb_ + row_offset];
            } else {
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
                } else {
                    // add proper zero padding
                    b_val[j] = 0;
                    // do not increment pointer for padding
                }
            }
            // write to the pack buffer all at once
            for (j = 0; j < RN; j++) {
                *packed_B++ = b_val[j];
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

        TA *packA = malloc_aligned<TA>(CK, (CM + 1) * ic_nts, sizeof(TA));
        TB *packB = malloc_aligned<TB>(CK, CN + 1, sizeof(TB));

        // 5-th loop around micro-kernel
        for (jc = 0; jc < n; jc += CN) {
            jb = std::min(n - jc, CN);
            // 4-th loop around micro-kernel
            for (pc = 0; pc < k; pc += CK) {
                pb = std::min(k - pc, CK);

                // packing sub-matrix B
                #pragma omp parallel for num_threads(ic_nts)
                for (j = 0; j < jb; j += RN) {
                    pack_matrix_B(pb,                    // number of rows to actually pack
                                  std::min(jb - j, RN),  // number of columns to actually pack
                                  pc,                    // global row offset
                                  jc + j,                // global columns offset
                                  B_,                    // original matrix pointer
                                  &packB[j * pb]         // packed buffer
                    );
                }

                // start a parallel region
                #pragma omp parallel num_threads(ic_nts) private(ic, ib, i)
                {
                    int64_t thread_id = omp_get_thread_num();
                    // call partition_workload_by_thread
                    int64_t range_start, range_end;
                    partition_workload_by_thread(m, RM, range_start, range_end);

                    // 3-rd loop around micro-kernel
                    for (ic = range_start; ic < range_end; ic += CM) {
                        ib = std::min(range_end - ic, CM);
                        // packing sub-matrix A
                        for (i = 0; i < ib; i += RM) {
                            pack_matrix_A(
                                std::min(ib - i, RM),  // number of rows to actually pack
                                pb,                    // umber of columns to actually pack
                                ic + i,                // global row offset
                                pc,                    // global columns offset
                                A_,                    // original matrix pointer
                                &packA[thread_id * CM * pb + i * pb]  // packed buffer position
                            );
                        }

                        // define micro-kernel ctx
                        MicroKernelCtxType<TB> ctx;
                        ctx.next = packB;

                        // 2-th loop around micro-kernel
                        for (j = 0; j < jb; j += RN) {
                            ctx.n = std::min(jb - j, RN);
                            for (i = 0; i < ib; i += RM) {
                                ctx.m = std::min(ib - i, RM);
                                if (i + RM > ib) {
                                    ctx.next += pb * RN;
                                }
                                micro_kernel_(pb,
                                              &packA[thread_id * CM * pb + i * pb],
                                              &packB[j * pb],
                                              &C_[(jc + j) * ldc_ + (ic + i)],
                                              ldc_, 
                                              &ctx);
                            }
                        }
                    }
                }
            }
        }
        free(packA);
        free(packB);
    };
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // TINYBLAS_GEMM_HPP_