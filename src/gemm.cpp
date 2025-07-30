#include <gemm.hpp>

namespace gemm {

template<int64_t RM, int64_t RN>
void matmul(int64_t m, int64_t n, int64_t k, 
           const float *A, int64_t lda,
           const float *B, int64_t ldb, 
           float *C, int64_t ldc) {
    
    // gemm::detail::GEMM<float, float, float, 4, 4> gemm_engine(A, lda, B, ldb, C, ldc);
    // gemm_engine.multiply(m, n, k);
    if constexpr (RM == 4 && RN == 4) {
        // 使用4x4内核
        gemm::detail::GEMM<float, float, float, 4, 4> 
            gemm_engine(A, lda, B, ldb, C, ldc, gemm::detail::AddDot_4x4_kernel_float<4, 4>);
        gemm_engine.multiply(m, n, k);
    }
    else if constexpr (RM == 8 && RN == 4) {
        // 使用8x4内核
        gemm::detail::GEMM<float, float, float, 8, 4> 
            gemm_engine(A, lda, B, ldb, C, ldc, gemm::detail::AddDot_8x4_kernel_float<8, 4>);
        gemm_engine.multiply(m, n, k);
    }
    else {
        // 默认使用4x4内核
        gemm::detail::GEMM<float, float, float, 4, 4> 
            gemm_engine(A, lda, B, ldb, C, ldc, gemm::detail::AddDot_4x4_kernel_float<4, 4>);
        gemm_engine.multiply(m, n, k);
    }
}

template<int64_t RM, int64_t RN>
void matmul(int64_t m, int64_t n, int64_t k,
           const double *A, int64_t lda,
           const double *B, int64_t ldb,
           double *C, int64_t ldc) {
    
    if constexpr (RM == 4 && RN == 4) {
        // 对于double类型，主要使用4x4内核
        gemm::detail::GEMM<double, double, double, 4, 4> 
            gemm_engine(A, lda, B, ldb, C, ldc, gemm::detail::AddDot_4x4_kernel_double<4, 4>);
        gemm_engine.multiply(m, n, k);
    }
    else if constexpr (RM == 8 && RN == 4) {
        // double类型的8x4内核（如果实现了的话）
        gemm::detail::GEMM<double, double, double, 8, 4> 
            gemm_engine(A, lda, B, ldb, C, ldc, gemm::detail::AddDot_8x4_kernel_double<8, 4>);
        gemm_engine.multiply(m, n, k);
    }
    else {
        // 默认使用4x4内核
        gemm::detail::GEMM<double, double, double, 4, 4> 
            gemm_engine(A, lda, B, ldb, C, ldc, gemm::detail::AddDot_4x4_kernel_double<4, 4>);
        gemm_engine.multiply(m, n, k);
    }
}

template void matmul<4, 4>(int64_t, int64_t, int64_t, 
                          const float*, int64_t, const float*, int64_t, float*, int64_t);
template void matmul<8, 4>(int64_t, int64_t, int64_t, 
                          const float*, int64_t, const float*, int64_t, float*, int64_t);

template void matmul<4, 4>(int64_t, int64_t, int64_t, 
                          const double*, int64_t, const double*, int64_t, double*, int64_t);

template void matmul<8, 4>(int64_t, int64_t, int64_t, 
                          const double*, int64_t, const double*, int64_t, double*, int64_t);

} // gemm
