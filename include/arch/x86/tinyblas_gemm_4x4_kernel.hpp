#ifndef X86_TINYBLAS_GEMM_4X4_KERNEL_HPP_
#define X86_TINYBLAS_GEMM_4X4_KERNEL_HPP_

#include <arch/tinyblas_kernels.hpp>

namespace tinyBLAS {
namespace detail {

template <int64_t RM, int64_t RN>
void AddDot_4x4_kernel_float(
    int64_t k, float *a, float *b, float *c, int64_t ldc, MicroKernelCtxType<float> *ctx) {
    int64_t p;
    float *b_next = ctx->next;
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
    __asm__ volatile("prefetcht0 2(%0)          \n\t" : : "r"(b_next));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(c));

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
    col01_lo_xmm = unpacklo(c_col0_xmm, c_col1_xmm);  // [c_col0[0], c_col1[0], c_col0[1], c_col1[1]]
    col23_lo_xmm = unpacklo(c_col2_xmm, c_col3_xmm);  // [c_col2[0], c_col3[0], c_col2[1], c_col3[1]]
    col01_hi_xmm = unpackhi(c_col0_xmm, c_col1_xmm);  // [c_col0[2], c_col1[2], c_col0[3], c_col1[3]]
    col23_hi_xmm = unpackhi(c_col2_xmm, c_col3_xmm);  // [c_col2[2], c_col3[2], c_col2[3], c_col3[3]]

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
    float *c_ptr = c;
    store(c_ptr, add(mul(beta_xmm, load<__m128>(c_ptr)), mul(alpha_xmm, c_row0_xmm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_xmm, load<__m128>(c_ptr)), mul(alpha_xmm, c_row1_xmm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_xmm, load<__m128>(c_ptr)), mul(alpha_xmm, c_row2_xmm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_xmm, load<__m128>(c_ptr)), mul(alpha_xmm, c_row3_xmm)));
}

template <int64_t RM, int64_t RN>
void AddDot_4x4_kernel_double(
    int64_t k, double *a, double *b, double *c, int64_t ldc, MicroKernelCtxType<double> *ctx) {
    int64_t p;
    double *b_next = ctx->next;
    double alpha = 1.0, beta = 1.0;
    
    // define vars
    double *c_row0, *c_row1, *c_row2, *c_row3;
    __m256d c_row0_orig_ymm, c_row1_orig_ymm, c_row2_orig_ymm, c_row3_orig_ymm;
    __m256d c_row0_ymm, c_row1_ymm, c_row2_ymm, c_row3_ymm;
    __m256d tmp_ymm;

    // init accumulator, each accumulator corresponds to one column of matrix C
    __m256d c_col0_ymm = setzeros<__m256d>();
    __m256d c_col1_ymm = setzeros<__m256d>();
    __m256d c_col2_ymm = setzeros<__m256d>();
    __m256d c_col3_ymm = setzeros<__m256d>();

    // pre-fetch data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(a));
    __asm__ volatile("prefetcht0 2(%0)          \n\t" : : "r"(b_next));
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
        __asm__ volatile("prefetcht0 256(%0)          \n\t" : : "r"(a));
        __asm__ volatile("prefetcht0 256(%0)          \n\t" : : "r"(b));
        
        // pre-load a for next k = 2 
        a_col_ymm1 = load<__m256d>(a + 4);
        
        // handle a[0:4] * b[0] and a_ik * b[1] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm0, 0x00);
        c_col0_ymm = madd(a_col_ymm0, tmp_ymm, c_col0_ymm);
        tmp_ymm = permute4x64(b_row_ymm0, 0x55);
        c_col1_ymm = madd(a_col_ymm0, tmp_ymm, c_col1_ymm);

        // pre-load b for next k = 2 
        b_row_ymm1 = load<__m256d>(b + 4);

        // handle a[0:4] * b[2] and a_ik * b[3] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm0, 0xAA);
        c_col2_ymm = madd(a_col_ymm0, tmp_ymm, c_col2_ymm);
        tmp_ymm = permute4x64(b_row_ymm0, 0xFF);
        c_col3_ymm = madd(a_col_ymm0, tmp_ymm, c_col3_ymm);

        // prefetch data
        __asm__ volatile("prefetcht0 512(%0)         \n\t" : : "r"(a));
        __asm__ volatile("prefetcht1 512(%0)         \n\t" : : "r"(b));

        // pre-load A for next iteration
        a_col_ymm0 = load<__m256d>(a + 8);
        
        // handle a[0:4] * b[0] and a_ik * b[1] for 2ed iteration
        tmp_ymm = permute4x64(b_row_ymm1, 0x00);
        c_col0_ymm = madd(a_col_ymm1, tmp_ymm, c_col0_ymm);
        tmp_ymm = permute4x64(b_row_ymm1, 0x55);
        c_col1_ymm = madd(a_col_ymm1, tmp_ymm, c_col1_ymm);

        // pre-load B for next iteration
        b_row_ymm0 = load<__m256d>(b + 8);

        // handle a[0:4] * b[2] and a_ik * b[3] for 2ed iteration
        tmp_ymm = permute4x64(b_row_ymm1, 0xAA);
        c_col2_ymm = madd(a_col_ymm1, tmp_ymm, c_col2_ymm);
        tmp_ymm = permute4x64(b_row_ymm1, 0xFF);
        c_col3_ymm = madd(a_col_ymm1, tmp_ymm, c_col3_ymm);

        // update pointer
        a += 8;
        b += 8;
    }

    // handle remaining elements
    for (p = 0; p < k_remainder; ++p) {
        __m256d a_col_ymm = load<__m256d>(a);
        __m256d b_row_ymm = load<__m256d>(b);

        tmp_ymm = permute4x64(b_row_ymm, 0x00);
        c_col0_ymm = madd(a_col_ymm, tmp_ymm, c_col0_ymm);
        tmp_ymm = permute4x64(b_row_ymm, 0x55);
        c_col1_ymm = madd(a_col_ymm, tmp_ymm, c_col1_ymm);
        tmp_ymm = permute4x64(b_row_ymm, 0xAA);
        c_col2_ymm = madd(a_col_ymm, tmp_ymm, c_col2_ymm);
        tmp_ymm = permute4x64(b_row_ymm, 0xFF);
        c_col3_ymm = madd(a_col_ymm, tmp_ymm, c_col3_ymm);

        // update pointer
        a += 4;
        b += 4;
    }

    // transpose a 4x4 matrix: convert from column-major order to row-major order
    __m128d c_col0_lo_xmm = castpd256(c_col0_ymm);        // col0[0:1]
    __m128d c_col0_hi_xmm = extractf128(c_col0_ymm, 1);   // col0[2:3]
    __m128d c_col1_lo_xmm = castpd256(c_col1_ymm);        // col1[0:1]
    __m128d c_col1_hi_xmm = extractf128(c_col1_ymm, 1);   // col1[2:3]
    __m128d c_col2_lo_xmm = castpd256(c_col2_ymm);        // col2[0:1]
    __m128d c_col2_hi_xmm = extractf128(c_col2_ymm, 1);   // col2[2:3]
    __m128d c_col3_lo_xmm = castpd256(c_col3_ymm);        // col3[0:1]
    __m128d c_col3_hi_xmm = extractf128(c_col3_ymm, 1);   // col3[2:3]

    // lower-128 
    __m128d c_row0_lo_xmm = unpacklo(c_col0_lo_xmm, c_col1_lo_xmm); // col0[0], col1[0]
    __m128d c_row0_hi_xmm = unpacklo(c_col2_lo_xmm, c_col3_lo_xmm); // col2[0], col3[0]
    __m128d c_row1_lo_xmm = unpackhi(c_col0_lo_xmm, c_col1_lo_xmm); // col0[1], col1[1]
    __m128d c_row1_hi_xmm = unpackhi(c_col2_lo_xmm, c_col3_lo_xmm); // col2[1], col3[1]

    // high-128
    __m128d c_row2_lo_xmm = unpacklo(c_col0_hi_xmm, c_col1_hi_xmm); // col0[2], col1[2]
    __m128d c_row2_hi_xmm = unpacklo(c_col2_hi_xmm, c_col3_hi_xmm); // col2[2], col3[2]
    __m128d c_row3_lo_xmm = unpackhi(c_col0_hi_xmm, c_col1_hi_xmm); // col0[3], col1[3]
    __m128d c_row3_hi_xmm = unpackhi(c_col2_hi_xmm, c_col3_hi_xmm); // col2[3], col3[3]

    // set back to _m256d register from _m128d register
    c_row0_ymm = pack128(c_row0_hi_xmm, c_row0_lo_xmm); // [col0[0], col1[0], col2[0], col3[0]]
    c_row1_ymm = pack128(c_row1_hi_xmm, c_row1_lo_xmm); // [col0[1], col1[1], col2[1], col3[1]]
    c_row2_ymm = pack128(c_row2_hi_xmm, c_row2_lo_xmm); // [col0[2], col1[2], col2[2], col3[2]]
    c_row3_ymm = pack128(c_row3_hi_xmm, c_row3_lo_xmm); // [col0[3], col1[3], col2[3], col3[3]]

    __m256d alpha_ymm = set1<__m256d>(alpha);
    __m256d beta_ymm = set1<__m256d>(beta);

    // add the transposed result to matrix C
    double *c_ptr = c;
    store(c_ptr, add(mul(beta_ymm, load<__m256d>(c_ptr)), mul(alpha_ymm, c_row0_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256d>(c_ptr)), mul(alpha_ymm, c_row1_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256d>(c_ptr)), mul(alpha_ymm, c_row2_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256d>(c_ptr)), mul(alpha_ymm, c_row3_ymm)));
}

template <typename TA, typename TB, typename TC, int64_t RM, int64_t RN>
void AddDot_4x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc, MicroKernelCtxType<TB> *ctx) {
    if constexpr (std::is_same_v<TA, float> && std::is_same_v<TB, float> &&
                  std::is_same_v<TC, float>) {
        AddDot_4x4_kernel_float<RM, RN>(k, a, b, c, ldc, ctx);
    } else if constexpr (std::is_same_v<TA, double> && std::is_same_v<TB, double> &&
                         std::is_same_v<TC, double>) {
        AddDot_4x4_kernel_double<RM, RN>(k, a, b, c, ldc, ctx);
    }
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // X86_GEMM_4X4_KERNEL_HPP_