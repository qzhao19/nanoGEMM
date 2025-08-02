#include <gemm.hpp>
#include <stdexcept>

namespace gemm {
namespace detail {

template<>
void KernelFactory<float, float, float>::registerDefaultKernels() {
    // register kernel
    registerKernel<4, 4, 72, 256, 2048>("4x4", AddDot_4x4_kernel<float, float, float, 4, 4>);
    registerKernel<8, 4, 72, 256, 2048>("8x4", AddDot_8x4_kernel<float, float, float, 8, 4>);
}

template<>
void KernelFactory<double, double, double>::registerDefaultKernels() {
    registerKernel<4, 4, 72, 256, 2048>("4x4", AddDot_4x4_kernel<double, double, double, 4, 4>);
    registerKernel<8, 4, 72, 256, 2048>("8x4", AddDot_8x4_kernel<double, double, double, 8, 4>);
}

}

void matmul(int64_t m, int64_t n, int64_t k,
            const float *A, int64_t lda,
            const float *B, int64_t ldb,
            float *C, int64_t ldc, 
            const std::string &kernel) {
    
    auto& factory = detail::KernelFactory<float, float, float>::getInstance();
    
    try {
        auto executor = factory.createExecutor(kernel);
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
    }
    catch (const std::runtime_error& e) {
        #ifdef DEBUG
        throw std::invalid_argument("Kernel creation failed: " + std::string(e.what()));
        #else
        auto executor = factory.createExecutor("4x4");
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
        #endif
    }
}

void matmul(int64_t m, int64_t n, int64_t k,
            const double *A, int64_t lda,
            const double *B, int64_t ldb,
            double *C, int64_t ldc,
            const std::string &kernel) {
    
    auto& factory = detail::KernelFactory<double, double, double>::getInstance();
    
    try {
        auto executor = factory.createExecutor(kernel);
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
    }
    catch (const std::runtime_error& e) {
        #ifdef DEBUG
        throw std::invalid_argument("Kernel creation failed: " + std::string(e.what()));
        #else
        auto executor = factory.createExecutor("4x4");
        executor->multiply(m, n, k, A, lda, B, ldb, C, ldc);
        #endif
    }
}

}  // namespace gemm