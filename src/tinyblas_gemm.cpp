#include <tinyblas_gemm.hpp>

namespace tinyBLAS {
namespace detail {

template <>
void KernelFactory<float, float, float>::register_default_kernels() {
    // register kernel
    register_kernel<4, 4, 72, 256, 2048>(
        "4x4",
        [](int64_t k, const float* a, const float* b, float* c, int64_t ldc, MicroKernelCtxType<float>* ctx) {
            AddDot_4x4_kernel<float, float, float, 4, 4>(k, a, b, c, ldc, ctx);
        }
    );
    
    register_kernel<8, 8, 72, 256, 2048>(
        "8x8",
        [](int64_t k, const float* a, const float* b, float* c, int64_t ldc, MicroKernelCtxType<float>* ctx) {
            AddDot_8x8_kernel<float, float, float, 8, 8>(k, a, b, c, ldc, ctx);
        }
    );
}

template <>
void KernelFactory<double, double, double>::register_default_kernels() {
    register_kernel<4, 4, 72, 256, 2048>(
        "4x4",
        [](int64_t k, const double* a, const double* b, double* c, int64_t ldc, MicroKernelCtxType<double>* ctx) {
            AddDot_4x4_kernel<double, double, double, 4, 4>(k, a, b, c, ldc, ctx);
        }
    );
    
    register_kernel<8, 8, 72, 256, 2048>(
        "8x8",
        [](int64_t k, const double* a, const double* b, double* c, int64_t ldc, MicroKernelCtxType<double>* ctx) {
            AddDot_8x8_kernel<double, double, double, 8, 8>(k, a, b, c, ldc, ctx);
        }
    );
}

}  // namespace detail
}  // namespace tinyBLAS