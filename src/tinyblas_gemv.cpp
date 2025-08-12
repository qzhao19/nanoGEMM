#include <tinyblas_gemv.hpp>

namespace tinyBLAS {

inline void matmul(int64_t m, int64_t n,
                 const float* A, int64_t lda,
                 const float* x,
                 float* y) {

    detail::GEMVMicroKernelType<float, float, float, 32, 4> AddDot_32x4_kernel;
    tinyBLAS::detail::GEMV<float, float, float, 32, 4, 72, 256> gemv_engine(A, lda, x, y, AddDot_32x4_kernel);
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
