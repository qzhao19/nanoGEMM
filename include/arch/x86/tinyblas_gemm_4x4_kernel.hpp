#ifndef X86_TINYBLAS_GEMM_4X4_KERNEL_HPP_
#define X86_TINYBLAS_GEMM_4X4_KERNEL_HPP_

#include <arch/tinyblas_kernels.hpp>

namespace tinyBLAS {
namespace detail {

template <int64_t RM, int64_t RN>
void AddDot_4x4_kernel_float(int64_t k, float *a, float *b, float *c, int64_t ldc) {
    int64_t i;
    float alpha = 1.0f, beta = 1.0f;
    float *c0_3_0, *c0_3_1, *c0_3_2, *c0_3_3;
    
    // handle current data
    __m128 a0_3_xmm;
    // prefetch next data
    __m128 A0_3_xmm;
    __m128 B0_3_xmm;
    
    __m128 c0_3_0_xmm, c0_3_1_xmm, c0_3_2_xmm, c0_3_3_xmm;
    __m128 tmp_xmm;

    // accumulator
    __m128 a0_3b_0_xmm = setzeros<__m128>();
    __m128 a0_3b_1_xmm = setzeros<__m128>();
    __m128 a0_3b_2_xmm = setzeros<__m128>();
    __m128 a0_3b_3_xmm = setzeros<__m128>();

    // prefetch data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(a));
    __asm__ volatile("prefetcht2 0(%0)          \n\t" : :"r"(b));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(c));

    // load initializ row of matrix A
    a0_3_xmm = load<__m128>(a);
    
    // each iteration to process 2 batch data
    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;
    
    for (i = 0; i < k_blocks; ++i) {
        // prefetch data that is far from the current location
        __asm__ volatile("prefetcht0 64(%0)          \n\t" : :"r"(a));

        // pre-load data A of next k value
        A0_3_xmm = load<__m128>(a + 4);

        // handle 1st iteration k = 1, col = 0, 1
        tmp_xmm = set1<__m128>(b[0]);
        a0_3b_0_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_0_xmm);
        tmp_xmm = set1<__m128>(b[1]);
        a0_3b_1_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_1_xmm);

        // pre-load data B of k = 2
        B0_3_xmm = load<__m128>(b + 4);

        // handle 1st iteration k = 1, col = 2, 3
        tmp_xmm = set1<__m128>(b[2]);
        a0_3b_2_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_2_xmm);
        tmp_xmm = set1<__m128>(b[3]);
        a0_3b_3_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_3_xmm);

        // prefetch data that is more far
        __asm__ volatile("prefetcht0 128(%0)          \n\t" : :"r"(a));

        // pre-load A for next iteration
        a0_3_xmm = load<__m128>(a + 8);

        // handle 2ed iteration k = 2, col = 0, 1, 2, 4
        tmp_xmm = set1<__m128>(B0_3_xmm[0]);
        a0_3b_0_xmm = madd<__m128>(A0_3_xmm, tmp_xmm, a0_3b_0_xmm);
        tmp_xmm = set1<__m128>(B0_3_xmm[1]);
        a0_3b_1_xmm = madd<__m128>(A0_3_xmm, tmp_xmm, a0_3b_1_xmm);
        tmp_xmm = set1<__m128>(B0_3_xmm[2]);
        a0_3b_2_xmm = madd<__m128>(A0_3_xmm, tmp_xmm, a0_3b_2_xmm);
        tmp_xmm = set1<__m128>(B0_3_xmm[3]);
        a0_3b_3_xmm = madd<__m128>(A0_3_xmm, tmp_xmm, a0_3b_3_xmm);

        // update pointer
        a += 8;
        b += 8;
    }
    
    for (i = 0; i < k_remainder; ++i) {
        // compute the remaining k value
        a0_3_xmm = load<__m128>(a);

        // col = 0, 1, 2, 4
        tmp_xmm = set1<__m128>(b[0]);
        a0_3b_0_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_0_xmm);
        tmp_xmm = set1<__m128>(b[1]);
        a0_3b_1_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_1_xmm);
        tmp_xmm = set1<__m128>(b[2]);
        a0_3b_2_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_2_xmm);
        tmp_xmm = set1<__m128>(b[3]);
        a0_3b_3_xmm = madd<__m128>(a0_3_xmm, tmp_xmm, a0_3b_3_xmm);

        // update pointer
        a += 4;
        b += 4;
    }

    __m128 alpha_xmm = set1<__m128>(alpha);
    __m128 beta_xmm = set1<__m128>(beta);

    // Store results
    c0_3_0 = c + 0 * ldc;
    c0_3_0_xmm = load<__m128>(c0_3_0);
    c0_3_0_xmm = add(mul(beta_xmm, c0_3_0_xmm), mul(alpha_xmm, a0_3b_0_xmm));
    store(c0_3_0, c0_3_0_xmm);

    c0_3_1 = c + 1 * ldc;
    c0_3_1_xmm = load<__m128>(c0_3_1);
    c0_3_1_xmm = add(mul(beta_xmm, c0_3_1_xmm), mul(alpha_xmm, a0_3b_1_xmm));
    store(c0_3_1, c0_3_1_xmm);

    c0_3_2 = c + 2 * ldc;
    c0_3_2_xmm = load<__m128>(c0_3_2);
    c0_3_2_xmm = add(mul(beta_xmm, c0_3_2_xmm), mul(alpha_xmm, a0_3b_2_xmm));
    store(c0_3_2, c0_3_2_xmm);

    c0_3_3 = c + 3 * ldc;
    c0_3_3_xmm = load<__m128>(c0_3_3);
    c0_3_3_xmm = add(mul(beta_xmm, c0_3_3_xmm), mul(alpha_xmm, a0_3b_3_xmm));
    store(c0_3_3, c0_3_3_xmm);
};


template <int64_t RM, int64_t RN>
void AddDot_4x4_kernel_double(int64_t k, double *a, double *b, double *c, int64_t ldc) {
    int64_t i;
    double alpha = 1.0, beta = 1.0;
    double *c0_3_0, *c0_3_1, *c0_3_2, *c0_3_3;

    // define matrix a0_3 
    __m256d a0_3_ymm;
    
    // prefetch next data
    __m256d A0_3_ymm;
    __m256d B0_3_ymm;

    __m256d c0_3_0_ymm, c0_3_1_ymm, c0_3_2_ymm, c0_3_3_ymm;
    __m256d tmp_ymm;

    // accumulator
    __m256d a0_3b_0_ymm = setzeros<__m256d>();
    __m256d a0_3b_1_ymm = setzeros<__m256d>();
    __m256d a0_3b_2_ymm = setzeros<__m256d>();
    __m256d a0_3b_3_ymm = setzeros<__m256d>();

    // pre-fetch a, b, and c data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(a));
	__asm__ volatile("prefetcht2 0(%0)          \n\t" : :"r"(b));
	__asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(c));

    // load initializ row of matrix A
    a0_3_ymm = load<__m256d>(a);

    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;

    for (i = 0; i < k_blocks; ++i) {
        // prefetch data that is far from the current location
        __asm__ volatile("prefetcht0 128(%0)          \n\t" : :"r"(a));

        // pre-load data A of next k value
        A0_3_ymm = load<__m256d>(a + 4);

        // compute 1st k value, col = 1, 2
        tmp_ymm = set1<__m256d>(b[0]);
        a0_3b_0_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_0_ymm);
        tmp_ymm = set1<__m256d>(b[1]);
        a0_3b_1_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_1_ymm);

        // pre-load data B of k = 2
        B0_3_ymm = load<__m256d>(b + 4);
        
        // handle 1st iteration k = 1, col = 2, 3
        tmp_ymm = set1<__m256d>(b[2]);
        a0_3b_2_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_2_ymm);
        tmp_ymm = set1<__m256d>(b[3]);
        a0_3b_3_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_3_ymm);

        // prefetch data that is more far
        __asm__ volatile("prefetcht0 256(%0)          \n\t" : :"r"(a));

        // pre-load A for next iteration
        a0_3_ymm = load<__m256d>(a + 8);

        // handle 1st iteration k = 1, col = 1, 2, 3, 4
        tmp_ymm = set1<__m256d>(B0_3_ymm[0]);
        a0_3b_0_ymm = madd<__m256d>(A0_3_ymm, tmp_ymm, a0_3b_0_ymm);
        tmp_ymm = set1<__m256d>(B0_3_ymm[1]);
        a0_3b_1_ymm = madd<__m256d>(A0_3_ymm, tmp_ymm, a0_3b_1_ymm);
        tmp_ymm = set1<__m256d>(B0_3_ymm[2]);
        a0_3b_2_ymm = madd<__m256d>(A0_3_ymm, tmp_ymm, a0_3b_2_ymm);
        tmp_ymm = set1<__m256d>(B0_3_ymm[3]);
        a0_3b_3_ymm = madd<__m256d>(A0_3_ymm, tmp_ymm, a0_3b_3_ymm);
        // update pointer
        a += 8;
        b += 8;
    }

    for (i = 0; i < k_remainder; ++i) {
        a0_3_ymm = load<__m256d>(a);
        
        // col = 0, 1, 2, 4
        tmp_ymm = set1<__m256d>(b[0]);
        a0_3b_0_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_0_ymm);
        tmp_ymm = set1<__m256d>(b[1]);
        a0_3b_1_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_1_ymm);
        tmp_ymm = set1<__m256d>(b[2]);
        a0_3b_2_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_2_ymm);
        tmp_ymm = set1<__m256d>(b[3]);
        a0_3b_3_ymm = madd<__m256d>(a0_3_ymm, tmp_ymm, a0_3b_3_ymm);
        // update pointer
        a += 4;
        b += 4;
    }

    __m256d alpha_ymm = set1<__m256d>(alpha);
    __m256d beta_ymm = set1<__m256d>(beta);

    // 1st column
    c0_3_0 = c + 0 * ldc;
    c0_3_0_ymm = load<__m256d>(c0_3_0);
    c0_3_0_ymm = add(mul(beta_ymm, c0_3_0_ymm), mul(alpha_ymm, a0_3b_0_ymm));
    store(c0_3_0, c0_3_0_ymm);

    // 2ed col
    c0_3_1 = c + 1 * ldc;
    c0_3_1_ymm = load<__m256d>(c0_3_1);
    c0_3_1_ymm = add(mul(beta_ymm, c0_3_1_ymm), mul(alpha_ymm, a0_3b_1_ymm));
    store(c0_3_1, c0_3_1_ymm);

    // 3rd col
    c0_3_2 = c + 2 * ldc;
    c0_3_2_ymm = load<__m256d>(c0_3_2);
    c0_3_2_ymm = add(mul(beta_ymm, c0_3_2_ymm), mul(alpha_ymm, a0_3b_2_ymm));
    store(c0_3_2, c0_3_2_ymm);

    // 4th col
    c0_3_3 = c + 3 * ldc;
    c0_3_3_ymm = load<__m256d>(c0_3_3);
    c0_3_3_ymm = add(mul(beta_ymm, c0_3_3_ymm), mul(alpha_ymm, a0_3b_3_ymm));
    store(c0_3_3, c0_3_3_ymm);
};


template <typename TA, typename TB, typename TC, int64_t RM, int64_t RN>
void AddDot_4x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc) {
    if constexpr (std::is_same_v<TA, float> && 
                  std::is_same_v<TB, float> && 
                  std::is_same_v<TC, float>) {
        AddDot_4x4_kernel_float<RM, RN>(k, a, b, c, ldc);
    }
    else if constexpr (std::is_same_v<TA, double> && 
                       std::is_same_v<TB, double> && 
                       std::is_same_v<TC, double>) {
        AddDot_4x4_kernel_double<RM, RN>(k, a, b, c, ldc);
    }
};

} // namespace detail
} // namespace tinyBLAS

#endif // X86_GEMM_4X4_KERNEL_HPP_