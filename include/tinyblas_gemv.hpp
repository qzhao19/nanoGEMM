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

public:
    GEMV(const TA *A,
         int64_t lda,
         const TX *x,
         TY *y, 
         GEMVMicroKernelType<TA, TX, TY, RM, RN> micro_kernel)
        : A_(A), lda_(lda), x_(x), y_(y), micro_kernel_(micro_kernel){};
    ~GEMV() = default;
    
    void multiply(int64_t m, int64_t n) {
        int64_t ic, ib, jc, jb;
        
        #pragma omp parallel for
        for (jc = 0; jc < n; jc += CN) {
            jb = std::min(jc + CN, n);

            for (ic = 0; ic < m; ic += CM) {
                ib = std::min(ic + CM, m);

                for (int64_t i = ic; i < ib; i += RM) {
                    const int64_t nrows = std::min(ib - i, RM);
                    TY sum[RM] = {0.0f};
                    
                    __m256 y_j_ymm[RM];
                    for (int64_t r = 0; r < nrows; ++r) {
                        y_j_ymm[r] = setzeros<__m256>();
                    }
                    
                    for (int64_t j = jc; j + 7 < jb; j += 8) {
                        __m256 x_j_ymm = load<__m256>(&x_[j]);
                        
                        for (int64_t r = 0; r < nrows; ++r) {
                            float temp_a[8];
                            for (int64_t k = 0; k < 8; ++k) {
                                temp_a[k] = A_[(j+k) * lda_ + (i+r)];
                            }
                            
                            __m256 a_rj_ymm = load<__m256>(temp_a);
                            y_j_ymm[r] = madd<__m256>(a_rj_ymm, x_j_ymm, y_j_ymm[r]);
                        }
                    }
                    
                    for (int64_t r = 0; r < nrows; ++r) {
                        sum[r] += hsum(y_j_ymm[r]);
                    }

                    __m128 y_j_xmm[RM];
                    for (int64_t r = 0; r < nrows; ++r) {
                        y_j_xmm[r] = setzeros<__m128>();
                    }
                    
                    int64_t j = jc;
                    while (j + 7 < jb) j += 8;
                    
                    for (; j + 3 < jb; j += 4) {
                        __m128 x_j_xmm = load<__m128>(&x_[j]);
                        
                        for (int64_t r = 0; r < nrows; ++r) {
                            float temp_a[4];
                            for (int64_t k = 0; k < 4; ++k) {
                                temp_a[k] = A_[(j+k) * lda_ + (i+r)];
                            }
                            
                            __m128 a_rj_xmm = load<__m128>(temp_a);
                            y_j_xmm[r] = madd<__m128>(a_rj_xmm, x_j_xmm, y_j_xmm[r]);
                        }
                    }
                    
                    for (int64_t r = 0; r < nrows; ++r) {
                        sum[r] += hsum(y_j_xmm[r]);
                    }

                    while (j + 7 < jb) j += 8;
                    while (j + 3 < jb) j += 4;
                    
                    for (; j < jb; ++j) {
                        for (int64_t r = 0; r < nrows; ++r) {
                            sum[r] += A_[j * lda_ + (i + r)] * x_[j];
                        }
                    }
                    
                    for (int64_t r = 0; r < nrows; ++r) {
                        y_[i + r] += sum[r];
                    }
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