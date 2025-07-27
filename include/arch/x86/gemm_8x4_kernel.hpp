#ifndef X86_GEMM_8X4_KERNEL_HPP_
#define X86_GEMM_8X4_KERNEL_HPP_

#include <arch/gemm_kernels.hpp>

namespace gemm {
namespace detail {

template <int64_t RM = 8, int64_t RN = 4>
void AddDot_8x4_kernel_float(int64_t k, float *a, float *b, float *c, int64_t ldc) {
    int64_t i;
    float alpha = 1.0f, beta = 1.0f;
    float *c0_7_0, *c0_7_1, *c0_7_2, *c0_7_3;

    // define matrix a0_7 and c0_7_0_4
    __m256 a0_7_ymm;
    __m256 c0_7_0_ymm, c0_7_1_ymm, c0_7_2_ymm, c0_7_3_ymm;

    // accumulator
    __m256 a0_7b_0_ymm = setzeros<__m256>();
    __m256 a0_7b_1_ymm = setzeros<__m256>();
    __m256 a0_7b_2_ymm = setzeros<__m256>();
    __m256 a0_7b_3_ymm = setzeros<__m256>();

    // pre-fetch a, b, and c data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(a));
    __asm__ volatile("prefetcht2 0(%0)          \n\t" : :"r"(b));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(c));

    // process blocks of 4 elements for better performance
    int64_t k_blocks = k / 4;
    int64_t k_remainder = k % 4;

    for (i = 0; i < k_blocks; ++i) {
        a0_7_ymm = load<__m256>(a);
        a0_7b_0_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[0]), a0_7b_0_ymm);
        a0_7b_1_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[1]), a0_7b_1_ymm);
        a0_7b_2_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[2]), a0_7b_2_ymm);
        a0_7b_3_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[3]), a0_7b_3_ymm);

        a0_7_ymm = load<__m256>(a + 8);
        a0_7b_0_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[4]), a0_7b_0_ymm);
        a0_7b_1_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[5]), a0_7b_1_ymm);
        a0_7b_2_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[6]), a0_7b_2_ymm);
        a0_7b_3_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[7]), a0_7b_3_ymm);

        a0_7_ymm = load<__m256>(a + 16);
        a0_7b_0_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[8]), a0_7b_0_ymm);
        a0_7b_1_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[9]), a0_7b_1_ymm);
        a0_7b_2_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[10]), a0_7b_2_ymm);
        a0_7b_3_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[11]), a0_7b_3_ymm);

        a0_7_ymm = load<__m256>(a + 24);
        a0_7b_0_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[12]), a0_7b_0_ymm);
        a0_7b_1_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[13]), a0_7b_1_ymm);
        a0_7b_2_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[14]), a0_7b_2_ymm);
        a0_7b_3_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[15]), a0_7b_3_ymm);
        a += 32;
        b += 16;
    }

    // handle remaining elements
    for (i = 0; i < k_remainder; ++i) {
        a0_7_ymm = load<__m256>(a);
        a0_7b_0_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[0]), a0_7b_0_ymm);
        a0_7b_1_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[1]), a0_7b_1_ymm);
        a0_7b_2_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[2]), a0_7b_2_ymm);
        a0_7b_3_ymm = madd<__m256>(a0_7_ymm, set1<__m256>(b[3]), a0_7b_3_ymm);
        a += 8;
        b += 4;
    }

    // store results
    __m256 alpha_ymm = set1<__m256>(alpha);
    __m256 beta_ymm = set1<__m256>(beta);

    c0_7_0 = c + 0 * ldc;
    c0_7_0_ymm = load<__m256>(c0_7_0);
    c0_7_0_ymm = add(mul(beta_ymm, c0_7_0_ymm), mul(alpha_ymm, a0_7b_0_ymm));
    store(c0_7_0, c0_7_0_ymm);

    c0_7_1 = c + 1 * ldc;
    c0_7_1_ymm = load<__m256>(c0_7_1);
    c0_7_1_ymm = add(mul(beta_ymm, c0_7_1_ymm), mul(alpha_ymm, a0_7b_1_ymm));
    store(c0_7_1, c0_7_1_ymm);

    c0_7_2 = c + 2 * ldc;
    c0_7_2_ymm = load<__m256>(c0_7_2);
    c0_7_2_ymm = add(mul(beta_ymm, c0_7_2_ymm), mul(alpha_ymm, a0_7b_2_ymm));
    store(c0_7_2, c0_7_2_ymm);

    c0_7_3 = c + 3 * ldc;
    c0_7_3_ymm = load<__m256>(c0_7_3);
    c0_7_3_ymm = add(mul(beta_ymm, c0_7_3_ymm), mul(alpha_ymm, a0_7b_3_ymm));
    store(c0_7_3, c0_7_3_ymm);
}

template <int64_t RM = 8, int64_t RN = 4>
void AddDot_8x4_kernel_double(int64_t k, double *a, double *b, double *c, int64_t ldc) {
    int64_t i;
    double alpha = 1.0, beta = 1.0;

    double *c0_3_0, *c0_3_1, *c0_3_2, *c0_3_3;
    double *c4_7_0, *c4_7_1, *c4_7_2, *c4_7_3;

    // define matrix a0_3, a4_7, b0_4 and c0_7_0_4
    __m256d b0_4_ymm;
    __m256d a0_3_ymm, a4_7_ymm;
    __m256d c0_3_0_ymm, c0_3_1_ymm, c0_3_2_ymm, c0_3_3_ymm;
    __m256d c4_7_0_ymm, c4_7_1_ymm, c4_7_2_ymm, c4_7_3_ymm;

    // define accumulator
    __m256d a0_3b_0_ymm = setzeros<__m256d>();
    __m256d a0_3b_1_ymm = setzeros<__m256d>();
    __m256d a0_3b_2_ymm = setzeros<__m256d>();
    __m256d a0_3b_3_ymm = setzeros<__m256d>();

    __m256d a4_7b_0_ymm = setzeros<__m256d>();
    __m256d a4_7b_1_ymm = setzeros<__m256d>();
    __m256d a4_7b_2_ymm = setzeros<__m256d>();
    __m256d a4_7b_3_ymm = setzeros<__m256d>();

    // pre-fetch a, b, and c data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(a));
	__asm__ volatile("prefetcht2 0(%0)          \n\t" : :"r"(b));
	__asm__ volatile("prefetcht0 0(%0)          \n\t" : :"r"(c));

    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;
    

    for (i = 0; i < k_blocks; ++i) {
        a0_3_ymm = load<__m256d>(a);
        a4_7_ymm = load<__m256d>(a + 4);
        b0_4_ymm = load<__m256d>(b);
        
        a0_3_ymm = madd<__m256d>(a0_3_ymm, b0_4_ymm, a0_3b_0_ymm);
        a4_7_ymm = madd<__m256d>(a4_7_ymm, b0_4_ymm, a0_3b_0_ymm);

    }


}

template <typename TA, typename TB, typename TC, int64_t RM = 8, int64_t RN = 4>
void AddDot_8x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc);

}
}
#endif // X86_GEMM_8X4_KERNEL_HPP_