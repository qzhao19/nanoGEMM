#include <tinyblas_gemv.hpp>

namespace tinyBLAS {

void matmul(int64_t m, int64_t n,
            const float* A, int64_t lda,
            const float* x,
            float* y) {

    tinyBLAS::detail::GEMV<float, float, float, 32, 4, 72, 256> gemv_engine(
        A, lda, x, y, 
        [](int64_t m, int64_t n, const float* A, int64_t lda, const float* x, float* y) {
            tinyBLAS::detail::AddDot_32x4_kernel_float<32, 4>(m, n, A, lda, x, y);
        });
    gemv_engine.multiply(m, n);
};

// inline void matmul(int64_t m, int64_t n,
//                  const double* A, int64_t lda,
//                  const double* x,
//                  double* y) {
//     tinyBLAS::detail::GEMV<double, double, double, 32, 4, 72, 256> gemv_engine(A, lda, x, y);
//     gemv_engine.multiply(m, n);
// };

}  // namespace tinyBLAS
