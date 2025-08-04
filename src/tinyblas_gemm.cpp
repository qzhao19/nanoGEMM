#include <tinyblas_gemm.hpp>

namespace tinyBLAS {
namespace detail {

template<>
void KernelFactory<float, float, float>::register_default_kernels() {
    // register kernel
    register_kernel<4, 4, 72, 256, 2048>("4x4", AddDot_4x4_kernel<float, float, float, 4, 4>);
    register_kernel<8, 4, 72, 256, 2048>("8x4", AddDot_8x4_kernel<float, float, float, 8, 4>);
};

template<>
void KernelFactory<double, double, double>::register_default_kernels() {
    register_kernel<4, 4, 72, 256, 2048>("4x4", AddDot_4x4_kernel<double, double, double, 4, 4>);
    register_kernel<8, 4, 72, 256, 2048>("8x4", AddDot_8x4_kernel<double, double, double, 8, 4>);
};

}  // namespace detail
}  // namespace tinyBLAS