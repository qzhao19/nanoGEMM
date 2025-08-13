#ifndef TINYBLAS_GEMV_HPP_
#define TINYBLAS_GEMV_HPP_

#include <algorithm>
#include <arch/tinyblas_kernels.hpp>
#include <arch/x86/tinyblas_gemv_32x4_kernel.hpp>
#include <core/kernel_factory.hpp>
#include <core/tinyblas_base.hpp>
#include <string>

namespace tinyBLAS {
namespace detail {

template <typename TA,
          typename TX,
          typename TY,
          int64_t RM,
          int64_t RN,
          int64_t CM,
          int64_t CN>
class GEMV {
private:
    const TA *const A_;
    const TX *const x_;
    TY *const y_;
    const int64_t lda_;
    
    GEMVMicroKernelType<TA, TX, TY, RM, RN> micro_kernel_;

    void pack_matrix_A(
        int64_t m, int64_t n, int64_t row_offset, int64_t col_offset, const TA *A, TA *packed_A) {
        
        int64_t i, j;
        const TA *a_ptr[RM];

        for (i = 0; i < RM; ++i) {
            if (i < m) {
                a_ptr[i] = &A[(col_offset + 0) * lda_ + (row_offset + i)];
            } else {
                a_ptr[i] = &A[(col_offset + 0) * lda_ + (row_offset + 0)];
            }
        }

        for (j = 0; j < n; ++j) {
            TA a_val[RM];
            for (i = 0; i < RM; ++i) {
                if (i < m) {
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
    }

    void pack_vector_x(
        int64_t n, int64_t col_offset, const TX *x, TX *packed_x) {
        
        int64_t j;
        for (j = 0; j < n; ++j) {
            if (j < CN) {
                packed_x[j] = x[col_offset + j];
            } else {
                packed_x[j] = static_cast<TX>(0);
            } 
        }
    }

public:
    GEMV(const TA *A,
         int64_t lda,
         const TX *x,
         TY *y, 
         GEMVMicroKernelType<TA, TX, TY, RM, RN> micro_kernel)
        : A_(A), lda_(lda), x_(x), y_(y), micro_kernel_(micro_kernel){};
    ~GEMV() = default;
    
    void multiply(int64_t m, int64_t n) {
        int64_t i, j;
        int64_t ic, jc;
        int64_t ib, jb;
        
        TA *packed_A = malloc_aligned<TA>(CN, RM, sizeof(TA));
        TX *packed_x = malloc_aligned<TX>(CN, 1, sizeof(TX));

        // perform block on columns axis
        for (jc = 0; jc < n; jc += CN) {
            jb = std::min(n - jc, CN);
            
            // pack jb elements, starting from the offset jc
            pack_vector_x(jb, jc, x_, packed_x);

            // perform block on row
            for (ic = 0; ic < m; ic += CM) {
                // current row block size
                ib = std::min(m - ic, CM);
                
                for (i = 0; i < ib; i += RM) {
                    pack_matrix_A(
                        std::min(ib - i, RM),
                        jb,
                        ic + i,
                        jc,
                        A_,
                        packed_A
                    );

                    // call micro-kernel function
                    micro_kernel_(
                        jb,
                        packed_A,
                        packed_x,
                        &y_[ic + i]
                    );
                }
            }
        }
    }
};

}  // namespace detail

void matmul(int64_t m, int64_t n,
            const float* A, int64_t lda,
            const float* x,
            float* y);

void matmul(int64_t m, int64_t n,
            const double* A, int64_t lda,
            const double* x,
            double* y);

}  // namespace tinyBLAS

#endif  // TINYBLAS_GEMM_HPP_