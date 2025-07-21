#include <gemm.hpp>

namespace gemm {

// void matmul(int64_t m, int64_t n, int64_t k, 
//            const float *A, int64_t lda,
//            const float *B, int64_t ldb, 
//            float *C, int64_t ldc) {
//     // Select the optimal register block size and implementation based on matrix dimensions
//     if (m % 16 == 0 && (m / 16 >= 6)) {
//         // large-size, use 16x4 kernel
//         gemm::detail::GEMM<float, float, float, 16, 4, 128, 256, 2048> 
//             gemm_engine(A, lda, B, ldb, C, ldc);
//         gemm_engine.multiply(m, n, k);
//     }
//     else if (m % 8 == 0) {
//         // For medium-sized matrices, use an 8x4 kernel
//         gemm::detail::GEMM<float, float, float, 8, 4, 96, 256, 1024> 
//             gemm_engine(A, lda, B, ldb, C, ldc);
//         gemm_engine.multiply(m, n, k);
//     }
//     else if (m % 4 == 0) {
//         // For small matrices, use a 4x4 kernel
//         gemm::detail::GEMM<float, float, float, 4, 4, 72, 256, 512> 
//             gemm_engine(A, lda, B, ldb, C, ldc);
//         gemm_engine.multiply(m, n, k);
//     }
//     else {
//         // By default, use a 4x4 kernel
//         gemm::detail::GEMM<float, float, float, 4, 4> 
//             gemm_engine(A, lda, B, ldb, C, ldc);
//         gemm_engine.multiply(m, n, k);
//     }
// };

// void matmul(int64_t m, int64_t n, int64_t k,
//            const double *A, int64_t lda,
//            const double *B, int64_t ldb,
//            double *C, int64_t ldc) {
//     if (m % 16 == 0 && (m / 16 >= 6)) {
//         gemm::detail::GEMM<double, double, double, 16, 4, 128, 256, 2048> 
//             gemm_engine(A, lda, B, ldb, C, ldc);
//         gemm_engine.multiply(m, n, k);
//     }
//     else if (m % 8 == 0) {
//         gemm::detail::GEMM<double, double, double, 8, 4, 96, 256, 1024> 
//             gemm_engine(A, lda, B, ldb, C, ldc);
//         gemm_engine.multiply(m, n, k);
//     }
//     else {
//         gemm::detail::GEMM<double, double, double, 4, 4> 
//             gemm_engine(A, lda, B, ldb, C, ldc);
//         gemm_engine.multiply(m, n, k);
//     }
// };

void matmul(int64_t m, int64_t n, int64_t k, 
           const float *A, int64_t lda,
           const float *B, int64_t ldb, 
           float *C, int64_t ldc) {
    
    gemm::detail::GEMM<float, float, float, 4, 4> gemm_engine(A, lda, B, ldb, C, ldc);
    gemm_engine.multiply(m, n, k);
}

void matmul(int64_t m, int64_t n, int64_t k,
           const double *A, int64_t lda,
           const double *B, int64_t ldb,
           double *C, int64_t ldc) {
    
    gemm::detail::GEMM<double, double, double, 4, 4> gemm_engine(A, lda, B, ldb, C, ldc);
    gemm_engine.multiply(m, n, k);
}

} // gemm
