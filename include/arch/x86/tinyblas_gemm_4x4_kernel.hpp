#ifndef X86_TINYBLAS_GEMM_4X4_KERNEL_HPP_
#define X86_TINYBLAS_GEMM_4X4_KERNEL_HPP_

#include <arch/tinyblas_kernels.hpp>

namespace tinyBLAS {
namespace detail {

template <int64_t RM, int64_t RN>
void AddDot_4x4_kernel_float(int64_t k, float *a, float *b, float *c, int64_t ldc) {
    int64_t p;
    // float *b_next = ctx->next;
    float alpha = 1.0f, beta = 1.0f;
    float *c_row0, *c_row1, *c_row2, *c_row3;

    __m128 c_row0_orig_xmm, c_row1_orig_xmm, c_row2_orig_xmm, c_row3_orig_xmm;
    __m128 col01_lo_xmm, col23_lo_xmm, col01_hi_xmm, col23_hi_xmm;
    __m128 c_row0_xmm, c_row1_xmm, c_row2_xmm, c_row3_xmm;
    __m128 tmp_xmm;

    // init accumulator, each accumulator corresponds to one column of matrix C
    __m128 c_col0_xmm = setzeros<__m128>();
    __m128 c_col1_xmm = setzeros<__m128>();
    __m128 c_col2_xmm = setzeros<__m128>();
    __m128 c_col3_xmm = setzeros<__m128>();

    // pre-fetch data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(a));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(b));
    __asm__ volatile("prefetcht2 0(%0)          \n\t" : : "r"(c));

    // each iteration to process 2 batch data
    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;

    __m128 a_col_xmm0, b_row_xmm0;
    __m128 a_col_xmm1, b_row_xmm1;

    // load initializ a and b
    a_col_xmm0 = load<__m128>(a);
    b_row_xmm0 = load<__m128>(b);

    // main loop
    for (p = 0; p < k_blocks; ++p) {
        // prefetch data that is far from the current location
        __asm__ volatile("prefetcht0 64(%0)          \n\t" : : "r"(a));
        __asm__ volatile("prefetcht0 64(%0)          \n\t" : : "r"(b));
        
        // pre-load a for next k = 2 
        a_col_xmm1 = load<__m128>(a + 4);
        
        // handle a[0:4] * b[0] and a_ik * b[1] for 1st iteration
        tmp_xmm = shuffle(b_row_xmm0, b_row_xmm0, 0x00);
        c_col0_xmm = madd(a_col_xmm0, tmp_xmm, c_col0_xmm);
        tmp_xmm = shuffle(b_row_xmm0, b_row_xmm0, 0x55);
        c_col1_xmm = madd(a_col_xmm0, tmp_xmm, c_col1_xmm);

        // pre-load b for next k = 2 
        b_row_xmm1 = load<__m128>(b + 4);

        // handle a[0:4] * b[2] and a_ik * b[3] for 1st iteration
        tmp_xmm = shuffle(b_row_xmm0, b_row_xmm0, 0xAA);
        c_col2_xmm = madd(a_col_xmm0, tmp_xmm, c_col2_xmm);
        tmp_xmm = shuffle(b_row_xmm0, b_row_xmm0, 0xFF);
        c_col3_xmm = madd(a_col_xmm0, tmp_xmm, c_col3_xmm);

        // prefetch data
        __asm__ volatile("prefetcht0 128(%0)         \n\t" : : "r"(a));
        __asm__ volatile("prefetcht1 128(%0)         \n\t" : : "r"(b));

        // pre-load A for next iteration
        a_col_xmm0 = load<__m128>(a + 8);
        
        // handle a[0:4] * b[0] and a_ik * b[1] for 2ed iteration
        tmp_xmm = shuffle(b_row_xmm1, b_row_xmm1, 0x00);
        c_col0_xmm = madd(a_col_xmm1, tmp_xmm, c_col0_xmm);
        tmp_xmm = shuffle(b_row_xmm1, b_row_xmm1, 0x55);
        c_col1_xmm = madd(a_col_xmm1, tmp_xmm, c_col1_xmm);

        // pre-load B for next iteration
        b_row_xmm0 = load<__m128>(b + 8);

        // handle a[0:4] * b[2] and a_ik * b[3] for 2ed iteration
        tmp_xmm = shuffle(b_row_xmm1, b_row_xmm1, 0xAA);
        c_col2_xmm = madd(a_col_xmm1, tmp_xmm, c_col2_xmm);
        tmp_xmm = shuffle(b_row_xmm1, b_row_xmm1, 0xFF);
        c_col3_xmm = madd(a_col_xmm1, tmp_xmm, c_col3_xmm);

        // update pointer
        a += 8;
        b += 8;
    }

    // handle remaining elements
    for (p = 0; p < k_remainder; ++p) {
        __m128 a_col_xmm = load<__m128>(a);
        __m128 b_row_xmm = load<__m128>(b);

        tmp_xmm = shuffle(b_row_xmm, b_row_xmm, 0x00);
        c_col0_xmm = madd(a_col_xmm, tmp_xmm, c_col0_xmm);
        tmp_xmm = shuffle(b_row_xmm, b_row_xmm, 0x55);
        c_col1_xmm = madd(a_col_xmm, tmp_xmm, c_col1_xmm);
        tmp_xmm = shuffle(b_row_xmm, b_row_xmm, 0xAA);
        c_col2_xmm = madd(a_col_xmm, tmp_xmm, c_col2_xmm);
        tmp_xmm = shuffle(b_row_xmm, b_row_xmm, 0xFF);
        c_col3_xmm = madd(a_col_xmm, tmp_xmm, c_col3_xmm);

        // update pointer
        a += 4;
        b += 4;
    }

    // transpose a 4x4 matrix: convert from column-major order to row-major order
    // [c_col0[0], c_col1[0], c_col0[1], c_col1[1]]
    // [c_col2[0], c_col3[0], c_col2[1], c_col3[1]]
    // [c_col0[2], c_col1[2], c_col0[3], c_col1[3]]
    // [c_col2[2], c_col3[2], c_col2[3], c_col3[3]]
    col01_lo_xmm = unpacklo(c_col0_xmm, c_col1_xmm);  
    col23_lo_xmm = unpacklo(c_col2_xmm, c_col3_xmm);  
    col01_hi_xmm = unpackhi(c_col0_xmm, c_col1_xmm);  
    col23_hi_xmm = unpackhi(c_col2_xmm, c_col3_xmm);  

    // [c_col0[0], c_col1[0], c_col2[0], c_col3[0]]
    // [c_col0[1], c_col1[1], c_col2[1], c_col3[1]]
    // [c_col0[2], c_col1[2], c_col2[2], c_col3[2]]
    // [c_col0[3], c_col1[3], c_col2[3], c_col3[3]]
    c_row0_xmm = shuffle(col01_lo_xmm, col23_lo_xmm, _MM_SHUFFLE(1, 0, 1, 0));  
    c_row1_xmm = shuffle(col01_lo_xmm, col23_lo_xmm, _MM_SHUFFLE(3, 2, 3, 2));  
    c_row2_xmm = shuffle(col01_hi_xmm, col23_hi_xmm, _MM_SHUFFLE(1, 0, 1, 0));  
    c_row3_xmm = shuffle(col01_hi_xmm, col23_hi_xmm, _MM_SHUFFLE(3, 2, 3, 2));  

    __m128 alpha_xmm = set1<__m128>(alpha);
    __m128 beta_xmm = set1<__m128>(beta);

    // add the transposed result to matrix C
    c_row0 = c + 0 * ldc;
    c_row1 = c + 1 * ldc;
    c_row2 = c + 2 * ldc;
    c_row3 = c + 3 * ldc;

    c_row0_orig_xmm = load<__m128>(c_row0);
    c_row1_orig_xmm = load<__m128>(c_row1);
    c_row2_orig_xmm = load<__m128>(c_row2);
    c_row3_orig_xmm = load<__m128>(c_row3);

    c_row0_orig_xmm = add(mul(beta_xmm, c_row0_orig_xmm), mul(alpha_xmm, c_row0_xmm));
    c_row1_orig_xmm = add(mul(beta_xmm, c_row1_orig_xmm), mul(alpha_xmm, c_row1_xmm));
    c_row2_orig_xmm = add(mul(beta_xmm, c_row2_orig_xmm), mul(alpha_xmm, c_row2_xmm));
    c_row3_orig_xmm = add(mul(beta_xmm, c_row3_orig_xmm), mul(alpha_xmm, c_row3_xmm));

    store(c_row0, c_row0_orig_xmm);
    store(c_row1, c_row1_orig_xmm);
    store(c_row2, c_row2_orig_xmm);
    store(c_row3, c_row3_orig_xmm);
}

template <int64_t RM, int64_t RN>
void AddDot_4x4_kernel_double(int64_t k, double *a, double *b, double *c, int64_t ldc) {
    int64_t p;
    // double *b_next = ctx->next;
    double alpha = 1.0, beta = 1.0;
    double *c_row0, *c_row1, *c_row2, *c_row3;

    __m256d c_row0_orig_ymm, c_row1_orig_ymm, c_row2_orig_ymm, c_row3_orig_ymm;
    __m256d col01_lo_ymm, col23_lo_ymm, col01_hi_ymm, col23_hi_ymm;
    __m256d c_row0_ymm, c_row1_ymm, c_row2_ymm, c_row3_ymm;
    __m256d tmp_ymm;

    // init accumulator, each accumulator corresponds to one column of matrix C
    __m256d c_col0_ymm = setzeros<__m256d>();
    __m256d c_col1_ymm = setzeros<__m256d>();
    __m256d c_col2_ymm = setzeros<__m256d>();
    __m256d c_col3_ymm = setzeros<__m256d>();

    // pre-fetch data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(a));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(b));
    __asm__ volatile("prefetcht2 0(%0)          \n\t" : : "r"(c));

    // each iteration to process 2 batch data
    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;

    __m256d a_col_ymm0, b_row_ymm0;
    __m256d a_col_ymm1, b_row_ymm1;

    // load initializ a and b
    a_col_ymm0 = load<__m256d>(a);
    b_row_ymm0 = load<__m256d>(b);

    // main loop
    for (p = 0; p < k_blocks; ++p) {
        // prefetch data that is far from the current location
        __asm__ volatile("prefetcht0 128(%0)          \n\t" : : "r"(a));
        __asm__ volatile("prefetcht0 128(%0)          \n\t" : : "r"(b));
        
        // pre-load a for next k = 2 
        a_col_ymm1 = load<__m256d>(a + 4);
        
        // handle a[0:4] * b[0] and a_ik * b[1] for 1st iteration
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm0, 0x00);
        c_col0_ymm = madd(a_col_ymm0, tmp_ymm, c_col0_ymm);
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm0, 0x55);
        c_col1_ymm = madd(a_col_ymm0, tmp_ymm, c_col1_ymm);

        // pre-load b for next k = 2 
        b_row_ymm1 = load<__m256d>(b + 4);

        // handle a[0:4] * b[2] and a_ik * b[3] for 1st iteration
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm0, 0xAA);
        c_col2_ymm = madd(a_col_ymm0, tmp_ymm, c_col2_ymm);
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm0, 0xFF);
        c_col3_ymm = madd(a_col_ymm0, tmp_ymm, c_col3_ymm);

        // prefetch data
        __asm__ volatile("prefetcht0 256(%0)         \n\t" : : "r"(a));
        __asm__ volatile("prefetcht1 256(%0)         \n\t" : : "r"(b));

        // pre-load A for next iteration
        a_col_ymm0 = load<__m256d>(a + 8);
        
        // handle a[0:4] * b[0] and a_ik * b[1] for 2ed iteration
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm1, 0x00);
        c_col0_ymm = madd(a_col_ymm1, tmp_ymm, c_col0_ymm);
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm1, 0x55);
        c_col1_ymm = madd(a_col_ymm1, tmp_ymm, c_col1_ymm);

        // pre-load B for next iteration
        b_row_ymm0 = load<__m256d>(b + 8);

        // handle a[0:4] * b[2] and a_ik * b[3] for 2ed iteration
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm1, 0xAA);
        c_col2_ymm = madd(a_col_ymm1, tmp_ymm, c_col2_ymm);
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm1, 0xFF);
        c_col3_ymm = madd(a_col_ymm1, tmp_ymm, c_col3_ymm);

        // update pointer
        a += 8;
        b += 8;
    }

    // handle remaining elements
    for (p = 0; p < k_remainder; ++p) {
        __m256d a_col_ymm = load<__m256d>(a);
        __m256d b_row_ymm = load<__m256d>(b);

        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm, 0x00);
        c_col0_ymm = madd(a_col_ymm, tmp_ymm, c_col0_ymm);
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm, 0x55);
        c_col1_ymm = madd(a_col_ymm, tmp_ymm, c_col1_ymm);
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm, 0xAA);
        c_col2_ymm = madd(a_col_ymm, tmp_ymm, c_col2_ymm);
        tmp_ymm = _mm256_permute4x64_pd(b_row_ymm, 0xFF);
        c_col3_ymm = madd(a_col_ymm, tmp_ymm, c_col3_ymm);

        // update pointer
        a += 4;
        b += 4;
    }

    // transpose a 4x4 matrix: convert from column-major order to row-major order
    // [c_col0[0], c_col1[0], c_col0[1], c_col1[1]]
    // [c_col2[0], c_col3[0], c_col2[1], c_col3[1]]
    // [c_col0[2], c_col1[2], c_col0[3], c_col1[3]]
    // [c_col2[2], c_col3[2], c_col2[3], c_col3[3]]
    col01_lo_ymm = unpacklo(c_col0_ymm, c_col1_ymm);  
    col23_lo_ymm = unpacklo(c_col2_ymm, c_col3_ymm);  
    col01_hi_ymm = unpackhi(c_col0_ymm, c_col1_ymm);  
    col23_hi_ymm = unpackhi(c_col2_ymm, c_col3_ymm);  

    // [c_col0[0], c_col1[0], c_col2[0], c_col3[0]]
    // [c_col0[1], c_col1[1], c_col2[1], c_col3[1]]
    // [c_col0[2], c_col1[2], c_col2[2], c_col3[2]]
    // [c_col0[3], c_col1[3], c_col2[3], c_col3[3]]
    c_row0_ymm = shuffle(col01_lo_ymm, col23_lo_ymm, _MM_SHUFFLE(1, 0, 1, 0));  
    c_row1_ymm = shuffle(col01_lo_ymm, col23_lo_ymm, _MM_SHUFFLE(3, 2, 3, 2));  
    c_row2_ymm = shuffle(col01_hi_ymm, col23_hi_ymm, _MM_SHUFFLE(1, 0, 1, 0));  
    c_row3_ymm = shuffle(col01_hi_ymm, col23_hi_ymm, _MM_SHUFFLE(3, 2, 3, 2));  

    __m256d alpha_ymm = set1<__m256d>(alpha);
    __m256d beta_ymm = set1<__m256d>(beta);

    // add the transposed result to matrix C
    c_row0 = c + 0 * ldc;
    c_row1 = c + 1 * ldc;
    c_row2 = c + 2 * ldc;
    c_row3 = c + 3 * ldc;

    c_row0_orig_ymm = load<__m256d>(c_row0);
    c_row1_orig_ymm = load<__m256d>(c_row1);
    c_row2_orig_ymm = load<__m256d>(c_row2);
    c_row3_orig_ymm = load<__m256d>(c_row3);

    c_row0_orig_ymm = add(mul(beta_ymm, c_row0_orig_ymm), mul(alpha_ymm, c_row0_ymm));
    c_row1_orig_ymm = add(mul(beta_ymm, c_row1_orig_ymm), mul(alpha_ymm, c_row1_ymm));
    c_row2_orig_ymm = add(mul(beta_ymm, c_row2_orig_ymm), mul(alpha_ymm, c_row2_ymm));
    c_row3_orig_ymm = add(mul(beta_ymm, c_row3_orig_ymm), mul(alpha_ymm, c_row3_ymm));

    store(c_row0, c_row0_orig_ymm);
    store(c_row1, c_row1_orig_ymm);
    store(c_row2, c_row2_orig_ymm);
    store(c_row3, c_row3_orig_ymm);
}


template <typename TA, typename TB, typename TC, int64_t RM, int64_t RN>
// void AddDot_4x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc, MicroKernelCtxType<TB> *ctx) {
void AddDot_4x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc) {

    if constexpr (std::is_same_v<TA, float> && std::is_same_v<TB, float> &&
                  std::is_same_v<TC, float>) {
        // AddDot_4x4_kernel_float<RM, RN>(k, a, b, c, ldc, ctx);
        AddDot_4x4_kernel_float<RM, RN>(k, a, b, c, ldc);
    } else if constexpr (std::is_same_v<TA, double> && std::is_same_v<TB, double> &&
                         std::is_same_v<TC, double>) {
        // AddDot_4x4_kernel_double<RM, RN>(k, a, b, c, ldc, ctx);
        AddDot_4x4_kernel_double<RM, RN>(k, a, b, c, ldc);
    }
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // X86_GEMM_4X4_KERNEL_HPP_