#ifndef X86_TINYBLAS_GEMM_8X4_KERNEL_HPP_
#define X86_TINYBLAS_GEMM_8X4_KERNEL_HPP_

#include <arch/tinyblas_kernels.hpp>

namespace tinyBLAS {
namespace detail {

template <int64_t RM, int64_t RN>
void AddDot_8x4_kernel_float(
    int64_t k, float *a, float *b, float *c, int64_t ldc, MicroKernelCtxType<float> *ctx) {
    int64_t i;
    float *b_next = ctx->next;
    float alpha = 1.0f, beta = 1.0f;
    float *c0_7_0, *c0_7_1, *c0_7_2, *c0_7_3;

    // current and prefetched data registers
    __m256 a0_7_ymm;  // currently used A data
    __m256 A0_7_ymm;  // prefetched next A block
    __m256 B0_3_ymm;  // prefetched next B block

    __m256 c0_7_0_ymm, c0_7_1_ymm, c0_7_2_ymm, c0_7_3_ymm;
    __m256 tmp_ymm;  // temporary calculation result

    // accumulators
    __m256 a0_7b_0_ymm = setzeros<__m256>();
    __m256 a0_7b_1_ymm = setzeros<__m256>();
    __m256 a0_7b_2_ymm = setzeros<__m256>();
    __m256 a0_7b_3_ymm = setzeros<__m256>();

    // prefetch initial data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(a));
    __asm__ volatile("prefetcht2 0(%0)          \n\t" : : "r"(b_next));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(c));

    // initial load
    a0_7_ymm = load<__m256>(a);

    // use double iteration
    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;

    for (i = 0; i < k_blocks; ++i) {
        // 192 bytes ahead of current A pointer
        __asm__ volatile("prefetcht0 192(%0)          \n\t" : : "r"(a));

        // prefetch next k value of A data
        A0_7_ymm = load<__m256>(a + 8);

        // first k value - 1, 2 column calculation
        tmp_ymm = set1<__m256>(b[0]);
        a0_7b_0_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_0_ymm);
        tmp_ymm = set1<__m256>(b[1]);
        a0_7b_1_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_1_ymm);

        // prefetch next B block
        B0_3_ymm = load<__m256>(b + 4);

        // first k value - 3rd, 4th column calculation
        tmp_ymm = set1<__m256>(b[2]);
        a0_7b_2_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_2_ymm);
        tmp_ymm = set1<__m256>(b[3]);
        a0_7b_3_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_3_ymm);

        // prefetch farther data (similar to 512 offset)
        __asm__ volatile("prefetcht0 512(%0)          \n\t" : : "r"(a));

        // load A data for next iteration
        a0_7_ymm = load<__m256>(a + 16);

        // second k value - use prefetched data
        // 1, 2, 3, 4 column calculation
        tmp_ymm = set1<__m256>(B0_3_ymm[0]);
        a0_7b_0_ymm = madd<__m256>(A0_7_ymm, tmp_ymm, a0_7b_0_ymm);
        tmp_ymm = set1<__m256>(B0_3_ymm[1]);
        a0_7b_1_ymm = madd<__m256>(A0_7_ymm, tmp_ymm, a0_7b_1_ymm);
        tmp_ymm = set1<__m256>(B0_3_ymm[2]);
        a0_7b_2_ymm = madd<__m256>(A0_7_ymm, tmp_ymm, a0_7b_2_ymm);
        tmp_ymm = set1<__m256>(B0_3_ymm[3]);
        a0_7b_3_ymm = madd<__m256>(A0_7_ymm, tmp_ymm, a0_7b_3_ymm);

        // update pointers - each iteration processes 2 k values, 2 k values * 8 elements
        a += 16;
        b += 8;
    }

    // handle remaining k values
    for (i = 0; i < k_remainder; ++i) {
        a0_7_ymm = load<__m256>(a);

        // column 1-4 calculations
        tmp_ymm = set1<__m256>(b[0]);
        a0_7b_0_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_0_ymm);
        tmp_ymm = set1<__m256>(b[1]);
        a0_7b_1_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_1_ymm);
        tmp_ymm = set1<__m256>(b[2]);
        a0_7b_2_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_2_ymm);
        tmp_ymm = set1<__m256>(b[3]);
        a0_7b_3_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7b_3_ymm);
        // update pointer
        a += 8;
        b += 4;
    }

    // store results
    __m256 alpha_ymm = set1<__m256>(alpha);
    __m256 beta_ymm = set1<__m256>(beta);

    c0_7_0 = c + 0 * ldc;
    c0_7_0_ymm = load<__m256>(c0_7_0);
    tmp_ymm = mul(alpha_ymm, a0_7b_0_ymm);
    c0_7_0_ymm = mul(beta_ymm, c0_7_0_ymm);
    c0_7_0_ymm = add(c0_7_0_ymm, tmp_ymm);
    store(c0_7_0, c0_7_0_ymm);

    c0_7_1 = c + 1 * ldc;
    c0_7_1_ymm = load<__m256>(c0_7_1);
    tmp_ymm = mul(alpha_ymm, a0_7b_1_ymm);
    c0_7_1_ymm = mul(beta_ymm, c0_7_1_ymm);
    c0_7_1_ymm = add(c0_7_1_ymm, tmp_ymm);
    store(c0_7_1, c0_7_1_ymm);

    c0_7_2 = c + 2 * ldc;
    c0_7_2_ymm = load<__m256>(c0_7_2);
    tmp_ymm = mul(alpha_ymm, a0_7b_2_ymm);
    c0_7_2_ymm = mul(beta_ymm, c0_7_2_ymm);
    c0_7_2_ymm = add(c0_7_2_ymm, tmp_ymm);
    store(c0_7_2, c0_7_2_ymm);

    c0_7_3 = c + 3 * ldc;
    c0_7_3_ymm = load<__m256>(c0_7_3);
    tmp_ymm = mul(alpha_ymm, a0_7b_3_ymm);
    c0_7_3_ymm = mul(beta_ymm, c0_7_3_ymm);
    c0_7_3_ymm = add(c0_7_3_ymm, tmp_ymm);
    store(c0_7_3, c0_7_3_ymm);
}

template <int64_t RM, int64_t RN>
void AddDot_8x4_kernel_double(
    int64_t k, double *a, double *b, double *c, int64_t ldc, MicroKernelCtxType<double> *ctx) {
    int64_t i;
    double alpha_val = 1.0, beta_val = 1.0;
    double *alpha, *beta;
    alpha = &alpha_val;
    beta = &beta_val;
    
    double *b_next = ctx->next;
    double *c0_3_0, *c0_3_1, *c0_3_2, *c0_3_3;
    double *c4_7_0, *c4_7_1, *c4_7_2, *c4_7_3;

    // define matrix a0_3, a4_7, b0_4 and result c0_7_0_4
    __m256d b0_ymm, b1_ymm, b2_ymm, b3_ymm;
    __m256d a0_3_ymm, a4_7_ymm;
    __m256d c0_3_0_ymm, c0_3_1_ymm, c0_3_2_ymm, c0_3_3_ymm;
    __m256d c4_7_0_ymm, c4_7_1_ymm, c4_7_2_ymm, c4_7_3_ymm;

    // temporary variables for prefetching and calculations
    __m256d A0_3_ymm, A4_7_ymm;
    __m256d B0_ymm;
    __m256d tmp_ymm;

    // define accumulator
    __m256d a0_3b_0_ymm = setzeros<__m256d>();
    __m256d a0_3b_1_ymm = setzeros<__m256d>();
    __m256d a0_3b_2_ymm = setzeros<__m256d>();
    __m256d a0_3b_3_ymm = setzeros<__m256d>();

    __m256d a4_7b_0_ymm = setzeros<__m256d>();
    __m256d a4_7b_1_ymm = setzeros<__m256d>();
    __m256d a4_7b_2_ymm = setzeros<__m256d>();
    __m256d a4_7b_3_ymm = setzeros<__m256d>();

    // Results after reorganization
    __m256d a0_3b0_ymm, a0_3b1_ymm, a0_3b2_ymm, a0_3b3_ymm;
    __m256d a4_7b0_ymm, a4_7b1_ymm, a4_7b2_ymm, a4_7b3_ymm;

    // Variables for blend operations
    __m256d tmp_a0_3b_0_ymm, tmp_a0_3b_1_ymm, tmp_a0_3b_2_ymm, tmp_a0_3b_3_ymm;
    __m256d tmp_a4_7b_0_ymm, tmp_a4_7b_1_ymm, tmp_a4_7b_2_ymm, tmp_a4_7b_3_ymm;

    // Alpha and beta vectors
    __m256d alpha_ymm, beta_ymm;

    // pre-fetch a, b, and c data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(a));
    __asm__ volatile("prefetcht2 0(%0)          \n\t" : : "r"(b_next));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(c));

    // Load initial data
    a0_3_ymm = load<__m256d>(a);
    a4_7_ymm = load<__m256d>(a + 4);
    b0_ymm = load<__m256d>(b);

    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;

    for (i = 0; i < k_blocks; ++i) {
        __asm__ volatile("prefetcht0 192(%0)          \n\t" : : "r"(a));

        // Prefetch next A data
        A0_3_ymm = load<__m256d>(a + 8);

        // Iteration 0: first column calculation
        a0_3b_0_ymm = madd<__m256d>(a0_3_ymm, b0_ymm, a0_3b_0_ymm);
        a4_7b_0_ymm = madd<__m256d>(a4_7_ymm, b0_ymm, a4_7b_0_ymm);

        // Prefetch next A data
        A4_7_ymm = load<__m256d>(a + 12);

        // Shuffle b for second column calculation: [b1,b0,b3,b2]
        b1_ymm = shuffle(b0_ymm, b0_ymm, 0x5);

        a0_3b_1_ymm = madd<__m256d>(a0_3_ymm, b1_ymm, a0_3b_1_ymm);
        a4_7b_1_ymm = madd<__m256d>(a4_7_ymm, b1_ymm, a4_7b_1_ymm);

        // Permute b for third column calculation: [b3,b2,b1,b0]
        b2_ymm = _mm256_permute2f128_pd(b1_ymm, b1_ymm, 0x1);

        // Prefetch next B data
        B0_ymm = load<__m256d>(b + 4);

        a0_3b_2_ymm = madd<__m256d>(a0_3_ymm, b2_ymm, a0_3b_2_ymm);
        a4_7b_2_ymm = madd<__m256d>(a4_7_ymm, b2_ymm, a4_7b_2_ymm);

        // Shuffle b for fourth column calculation: (b2,b3,b0,b1)
        b3_ymm = shuffle(b2_ymm, b2_ymm, 0x5);

        a0_3b_3_ymm = madd<__m256d>(a0_3_ymm, b3_ymm, a0_3b_3_ymm);
        a4_7b_3_ymm = madd<__m256d>(a4_7_ymm, b3_ymm, a4_7b_3_ymm);

        // Iteration 1
        __asm__ volatile("prefetcht0 512(%0)          \n\t" : : "r"(a));

        // Load next iteration's A data
        a0_3_ymm = load<__m256d>(a + 16);

        a0_3b_0_ymm = madd<__m256d>(A0_3_ymm, B0_ymm, a0_3b_0_ymm);

        // Shuffle B for next iteration
        b1_ymm = shuffle(B0_ymm, B0_ymm, 0x5);

        a4_7b_0_ymm = madd<__m256d>(A4_7_ymm, B0_ymm, a4_7b_0_ymm);
        a0_3b_1_ymm = madd<__m256d>(A0_3_ymm, b1_ymm, a0_3b_1_ymm);

        // Load next iteration's A data
        a4_7_ymm = load<__m256d>(a + 20);

        b2_ymm = _mm256_permute2f128_pd(b1_ymm, b1_ymm, 0x1);

        a4_7b_1_ymm = madd<__m256d>(A4_7_ymm, b1_ymm, a4_7b_1_ymm);
        a0_3b_2_ymm = madd<__m256d>(A0_3_ymm, b2_ymm, a0_3b_2_ymm);

        b3_ymm = shuffle(b2_ymm, b2_ymm, 0x5);
        a4_7b_2_ymm = madd<__m256d>(A4_7_ymm, b2_ymm, a4_7b_2_ymm);

        // Load next iteration's B data
        b0_ymm = load<__m256d>(b + 8);

        a0_3b_3_ymm = madd<__m256d>(A0_3_ymm, b3_ymm, a0_3b_3_ymm);
        a4_7b_3_ymm = madd<__m256d>(A4_7_ymm, b3_ymm, a4_7b_3_ymm);

        // Update pointers, move forward by 16 elements (8 for each iteration)
        a += 16;
        b += 8;
    }

    // Handle remaining k iterations
    for (i = 0; i < k_remainder; ++i) {
        // Load A data for this iteration
        a0_3_ymm = load<__m256d>(a);
        a4_7_ymm = load<__m256d>(a + 4);

        // Load B data for this iteration
        b0_ymm = load<__m256d>(b);
        // First column calculations
        a0_3b_0_ymm = madd<__m256d>(a0_3_ymm, b0_ymm, a0_3b_0_ymm);
        a4_7b_0_ymm = madd<__m256d>(a4_7_ymm, b0_ymm, a4_7b_0_ymm);

        // Shuffle b for second column: (b1,b0,b3,b2)
        b1_ymm = shuffle(b0_ymm, b0_ymm, 0x5);
        // Second column calculations
        a0_3b_1_ymm = madd<__m256d>(a0_3_ymm, b1_ymm, a0_3b_1_ymm);
        a4_7b_1_ymm = madd<__m256d>(a4_7_ymm, b1_ymm, a4_7b_1_ymm);

        // Permute b for third column: (b3,b2,b1,b0)
        b2_ymm = _mm256_permute2f128_pd(b1_ymm, b1_ymm, 0x1);
        // Third column calculations
        a0_3b_2_ymm = madd<__m256d>(a0_3_ymm, b2_ymm, a0_3b_2_ymm);
        a4_7b_2_ymm = madd<__m256d>(a4_7_ymm, b2_ymm, a4_7b_2_ymm);

        // Shuffle b for fourth column: (b2,b3,b0,b1)
        b3_ymm = shuffle(b2_ymm, b2_ymm, 0x5);

        // Fourth column calculations
        a0_3b_3_ymm = madd<__m256d>(a0_3_ymm, b3_ymm, a0_3b_3_ymm);
        a4_7b_3_ymm = madd<__m256d>(a4_7_ymm, b3_ymm, a4_7b_3_ymm);

        // Update pointers
        a += 8;  // Move forward by 8 elements
        b += 4;  // Move forward by 4 elements
    }

    // Reorganize results for proper storage
    beta_ymm = broadcast(beta);

    // Blend operations for reorganizing results
    tmp_a0_3b_0_ymm = _mm256_blend_pd(a0_3b_0_ymm, a0_3b_1_ymm, 0x6);  // 0110
    tmp_a0_3b_1_ymm = _mm256_blend_pd(a0_3b_1_ymm, a0_3b_0_ymm, 0x6);  // 0110

    tmp_a0_3b_2_ymm = _mm256_blend_pd(a0_3b_2_ymm, a0_3b_3_ymm, 0x6);  // 0110
    tmp_a0_3b_3_ymm = _mm256_blend_pd(a0_3b_3_ymm, a0_3b_2_ymm, 0x6);  // 0110

    tmp_a4_7b_0_ymm = _mm256_blend_pd(a4_7b_0_ymm, a4_7b_1_ymm, 0x6);  // 0110
    tmp_a4_7b_1_ymm = _mm256_blend_pd(a4_7b_1_ymm, a4_7b_0_ymm, 0x6);  // 0110

    tmp_a4_7b_2_ymm = _mm256_blend_pd(a4_7b_2_ymm, a4_7b_3_ymm, 0x6);  // 0110
    tmp_a4_7b_3_ymm = _mm256_blend_pd(a4_7b_3_ymm, a4_7b_2_ymm, 0x6);  // 0110

    alpha_ymm = broadcast(alpha);

    // Permute operations to finalize the reorganization
    a0_3b0_ymm =
        _mm256_permute2f128_pd(tmp_a0_3b_0_ymm, tmp_a0_3b_2_ymm, 0x30);  // 00|11|0000 = 0x30
    a0_3b3_ymm =
        _mm256_permute2f128_pd(tmp_a0_3b_2_ymm, tmp_a0_3b_0_ymm, 0x30);  // 00|11|0000 = 0x30

    a0_3b1_ymm =
        _mm256_permute2f128_pd(tmp_a0_3b_1_ymm, tmp_a0_3b_3_ymm, 0x30);  // 00|11|0000 = 0x30
    a0_3b2_ymm =
        _mm256_permute2f128_pd(tmp_a0_3b_3_ymm, tmp_a0_3b_1_ymm, 0x30);  // 00|11|0000 = 0x30

    a4_7b0_ymm =
        _mm256_permute2f128_pd(tmp_a4_7b_0_ymm, tmp_a4_7b_2_ymm, 0x30);  // 00|11|0000 = 0x30
    a4_7b3_ymm =
        _mm256_permute2f128_pd(tmp_a4_7b_2_ymm, tmp_a4_7b_0_ymm, 0x30);  // 00|11|0000 = 0x30

    a4_7b1_ymm =
        _mm256_permute2f128_pd(tmp_a4_7b_1_ymm, tmp_a4_7b_3_ymm, 0x30);  // 00|11|0000 = 0x30
    a4_7b2_ymm =
        _mm256_permute2f128_pd(tmp_a4_7b_3_ymm, tmp_a4_7b_1_ymm, 0x30);  // 00|11|0000 = 0x30

    // Store results back to matrix C (column-major format)
    // First column
    c0_3_0 = c;
    c0_3_0_ymm = load<__m256d>(c0_3_0);
    tmp_ymm = mul(alpha_ymm, a0_3b0_ymm);
    c0_3_0_ymm = madd(beta_ymm, c0_3_0_ymm, tmp_ymm);
    store(c0_3_0, c0_3_0_ymm);

    c4_7_0 = c + 4;
    c4_7_0_ymm = load<__m256d>(c4_7_0);
    tmp_ymm = mul(alpha_ymm, a4_7b0_ymm);
    c4_7_0_ymm = madd(beta_ymm, c4_7_0_ymm, tmp_ymm);
    store(c4_7_0, c4_7_0_ymm);

    // Second column
    c0_3_1 = c + ldc;
    c0_3_1_ymm = load<__m256d>(c0_3_1);
    tmp_ymm = mul(alpha_ymm, a0_3b1_ymm);
    c0_3_1_ymm = madd(beta_ymm, c0_3_1_ymm, tmp_ymm);
    store(c0_3_1, c0_3_1_ymm);

    c4_7_1 = c + ldc + 4;
    c4_7_1_ymm = load<__m256d>(c4_7_1);
    tmp_ymm = mul(alpha_ymm, a4_7b1_ymm);
    c4_7_1_ymm = madd(beta_ymm, c4_7_1_ymm, tmp_ymm);
    store(c4_7_1, c4_7_1_ymm);

    // Third column
    c0_3_2 = c + 2 * ldc;
    c0_3_2_ymm = load<__m256d>(c0_3_2);
    tmp_ymm = mul(alpha_ymm, a0_3b2_ymm);
    c0_3_2_ymm = madd(beta_ymm, c0_3_2_ymm, tmp_ymm);
    store(c0_3_2, c0_3_2_ymm);

    c4_7_2 = c + 2 * ldc + 4;
    c4_7_2_ymm = load<__m256d>(c4_7_2);
    tmp_ymm = mul(alpha_ymm, a4_7b2_ymm);
    c4_7_2_ymm = madd(beta_ymm, c4_7_2_ymm, tmp_ymm);
    store(c4_7_2, c4_7_2_ymm);

    // Fourth column
    c0_3_3 = c + 3 * ldc;
    c0_3_3_ymm = load<__m256d>(c0_3_3);
    tmp_ymm = mul(alpha_ymm, a0_3b3_ymm);
    c0_3_3_ymm = madd(beta_ymm, c0_3_3_ymm, tmp_ymm);
    store(c0_3_3, c0_3_3_ymm);

    c4_7_3 = c + 3 * ldc + 4;
    c4_7_3_ymm = load<__m256d>(c4_7_3);
    tmp_ymm = mul(alpha_ymm, a4_7b3_ymm);
    c4_7_3_ymm = madd(beta_ymm, c4_7_3_ymm, tmp_ymm);
    store(c4_7_3, c4_7_3_ymm);
}

template <typename TA, typename TB, typename TC, int64_t RM, int64_t RN>
void AddDot_8x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc, MicroKernelCtxType<TB> *ctx) {
    if constexpr (std::is_same_v<TA, float> && std::is_same_v<TB, float> &&
                  std::is_same_v<TC, float>) {
        AddDot_8x4_kernel_float<RM, RN>(k, a, b, c, ldc, ctx);
    } else if constexpr (std::is_same_v<TA, double> && std::is_same_v<TB, double> &&
                         std::is_same_v<TC, double>) {
        AddDot_8x4_kernel_double<RM, RN>(k, a, b, c, ldc, ctx);
    }
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // X86_GEMM_8X4_KERNEL_HPP_