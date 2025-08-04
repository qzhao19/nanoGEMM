#ifndef TINYBLAS_HPP_
#define TINYBLAS_HPP_

#include <tinyblas_gemm.hpp>

namespace tinyBLAS {

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

#endif // namespace tinyBLAS