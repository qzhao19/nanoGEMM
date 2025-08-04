#include <stdexcept>
#include <tinyblas.hpp>
#include <tinyblas_gemm.hpp>

namespace tinyBLAS {

void matmul(int64_t m, int64_t n, int64_t k,
            const float *A, int64_t lda,
            const float *B, int64_t ldb,
            float *C, int64_t ldc, 
            const std::string &kernel) {
    
    auto& factory = detail::KernelFactory<float, float, float>::get_instance();
    
    try {
        auto executor = factory.create_executor(kernel);
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
    }
    catch (const std::runtime_error& e) {
        #ifdef DEBUG
        throw std::invalid_argument("Kernel creation failed: " + std::string(e.what()));
        #else
        auto executor = factory.create_executor("4x4");
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
        #endif
    }
};

void matmul(int64_t m, int64_t n, int64_t k,
            const double *A, int64_t lda,
            const double *B, int64_t ldb,
            double *C, int64_t ldc,
            const std::string &kernel) {
    
    auto& factory = detail::KernelFactory<double, double, double>::get_instance();
    
    try {
        auto executor = factory.create_executor(kernel);
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
    }
    catch (const std::runtime_error& e) {
        #ifdef DEBUG
        throw std::invalid_argument("Kernel creation failed: " + std::string(e.what()));
        #else
        auto executor = factory.create_executor("4x4");
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
        #endif
    }
};

} // namespace tinyBLAS
