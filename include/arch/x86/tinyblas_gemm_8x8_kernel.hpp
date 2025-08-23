#ifndef X86_TINYBLAS_GEMM_8X8_KERNEL_HPP_
#define X86_TINYBLAS_GEMM_8X8_KERNEL_HPP_

#include <arch/tinyblas_kernels.hpp>

namespace tinyBLAS {
namespace detail {

template <int64_t RM, int64_t RN>
void AddDot_8x8_kernel_float(
    int64_t k, float *a, float *b, float *c, int64_t ldc, MicroKernelCtxType<float> *ctx) {
    int64_t i, p;
    float *b_next = ctx->next;
    float alpha = 1.0f, beta = 1.0f;
    
    // // init accumulator, each accumulator corresponds to one column of matrix C
    __m256 c_col0_ymm = _mm256_setzero_ps();
    __m256 c_col1_ymm = _mm256_setzero_ps();
    __m256 c_col2_ymm = _mm256_setzero_ps();
    __m256 c_col3_ymm = _mm256_setzero_ps();
    __m256 c_col4_ymm = _mm256_setzero_ps();
    __m256 c_col5_ymm = _mm256_setzero_ps();
    __m256 c_col6_ymm = _mm256_setzero_ps();
    __m256 c_col7_ymm = _mm256_setzero_ps();

    // pre-fetch data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(a));
    __asm__ volatile("prefetcht0 2(%0)          \n\t" : : "r"(b_next));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(c));

    // each iteration to process 2 batch data
    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;
    
    // define a0 a1 b0 b1, initializ a0 and b0
    __m256 a_col_ymm0, b_row_ymm0;
    __m256 a_col_ymm1, b_row_ymm1;
    __m256 tmp_ymm;
    a_col_ymm0 = load<__m256>(a);
    b_row_ymm0 = load<__m256>(b);

    for (p = 0; p < k_blocks; ++p) {
        // prefetch data that is far from the current location
        __asm__ volatile("prefetcht0 128(%0)          \n\t" : : "r"(a));
        __asm__ volatile("prefetcht0 128(%0)          \n\t" : : "r"(b));
        
        // pre-load a for next k = 2 
        a_col_ymm1 = load<__m256>(a + 8);

        // handle a[0:7] * b[0] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(0));
        c_col0_ymm = madd(a_col_ymm0, tmp_ymm, c_col0_ymm);
        // handle a[0:7] * b[1] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(1));
        c_col1_ymm = madd(a_col_ymm0, tmp_ymm, c_col1_ymm);
        // handle a[0:7] * b[2] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(2));
        c_col2_ymm = madd(a_col_ymm0, tmp_ymm, c_col2_ymm);
        // handle a[0:7] * b[3] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(3));
        c_col3_ymm = madd(a_col_ymm0, tmp_ymm, c_col3_ymm);

        // pre-load b for next k = 2 
        b_row_ymm1 = load<__m256>(b + 8);

        // handle a[0:7] * b[4] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(4));
        c_col4_ymm = madd(a_col_ymm0, tmp_ymm, c_col4_ymm);
        // handle a[0:7] * b[5] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(5));
        c_col5_ymm = madd(a_col_ymm0, tmp_ymm, c_col5_ymm);
        // handle a[0:7] * b[6] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(6));
        c_col6_ymm = madd(a_col_ymm0, tmp_ymm, c_col6_ymm);
        // handle a[0:7] * b[7] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm0, set1<__m256i>(7));
        c_col7_ymm = madd(a_col_ymm0, tmp_ymm, c_col7_ymm);

        // prefetch data
        __asm__ volatile("prefetcht0 256(%0)         \n\t" : : "r"(a));
        __asm__ volatile("prefetcht1 256(%0)         \n\t" : : "r"(b));

        // pre-load A for next iteration
        a_col_ymm0 = load<__m256>(a + 16);
        
        // handle a[8:15] * b[8] for 2ed iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(0));
        c_col0_ymm = madd(a_col_ymm1, tmp_ymm, c_col0_ymm);
        // handle a[8:15] * b[9] for 2ed iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(1));
        c_col1_ymm = madd(a_col_ymm1, tmp_ymm, c_col1_ymm);
        // handle a[8:15] * b[10] for 2ed iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(2));
        c_col2_ymm = madd(a_col_ymm1, tmp_ymm, c_col2_ymm);
        // handle a[8:15] * b[11] for 2ed iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(3));
        c_col3_ymm = madd(a_col_ymm1, tmp_ymm, c_col3_ymm);

        // pre-load B for next iteration
        b_row_ymm0 = load<__m256>(b + 16);

        // handle a[0:7] * b[4] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(4));
        c_col4_ymm = madd(a_col_ymm1, tmp_ymm, c_col4_ymm);
        // handle a[0:7] * b[5] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(5));
        c_col5_ymm = madd(a_col_ymm1, tmp_ymm, c_col5_ymm);
        // handle a[0:7] * b[6] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(6));
        c_col6_ymm = madd(a_col_ymm1, tmp_ymm, c_col6_ymm);
        // handle a[0:7] * b[7] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm1, set1<__m256i>(7));
        c_col7_ymm = madd(a_col_ymm1, tmp_ymm, c_col7_ymm);

        // update pointer
        a += 16;
        b += 16;
    }

    for (p = 0; p < k_remainder; ++p) {
        __m256 a_col_ymm = load<__m256>(a);
        __m256 b_row_ymm = load<__m256>(b);

        // handle a[0:7] * b[0] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(0));
        c_col0_ymm = madd(a_col_ymm, tmp_ymm, c_col0_ymm);
        // handle a[0:7] * b[1] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(1));
        c_col1_ymm = madd(a_col_ymm, tmp_ymm, c_col1_ymm);
        // handle a[0:7] * b[2] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(2));
        c_col2_ymm = madd(a_col_ymm, tmp_ymm, c_col2_ymm);
        // handle a[0:7] * b[3] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(3));
        c_col3_ymm = madd(a_col_ymm, tmp_ymm, c_col3_ymm);

        // handle a[0:7] * b[4] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(4));
        c_col4_ymm = madd(a_col_ymm, tmp_ymm, c_col4_ymm);
        // handle a[0:7] * b[5] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(5));
        c_col5_ymm = madd(a_col_ymm, tmp_ymm, c_col5_ymm);
        // handle a[0:7] * b[6] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(6));
        c_col6_ymm = madd(a_col_ymm, tmp_ymm, c_col6_ymm);
        // handle a[0:7] * b[7] for 1st iteration
        tmp_ymm = permutevar8x32(b_row_ymm, set1<__m256i>(7));
        c_col7_ymm = madd(a_col_ymm, tmp_ymm, c_col7_ymm);

        // update pointer
        a += 8;
        b += 8;
    }

    // matrix transposition: transpose 8 accumulated column vectors into 8 row vectors
    // unpack interleave pairs of columns
    __m256 col01_lo_ymm, col01_hi_ymm, col23_lo_ymm, col23_hi_ymm;
    __m256 col45_lo_ymm, col45_hi_ymm, col67_lo_ymm, col67_hi_ymm;
    col01_lo_ymm = unpacklo(c_col0_ymm, c_col1_ymm); // [c0[0],c1[0],c0[1],c1[1], c0[4],c1[4],c0[5],c1[5]]
    col01_hi_ymm = unpackhi(c_col0_ymm, c_col1_ymm); // [c0[2],c1[2],c0[3],c1[3], c0[6],c1[6],c0[7],c1[7]]
    col23_lo_ymm = unpacklo(c_col2_ymm, c_col3_ymm);
    col23_hi_ymm = unpackhi(c_col2_ymm, c_col3_ymm);
    col45_lo_ymm = unpacklo(c_col4_ymm, c_col5_ymm);
    col45_hi_ymm = unpackhi(c_col4_ymm, c_col5_ymm);
    col67_lo_ymm = unpacklo(c_col6_ymm, c_col7_ymm);
    col67_hi_ymm = unpackhi(c_col6_ymm, c_col7_ymm);

    // shuffle interleave 2-element groups
    __m256 row04_lo_ymm, row15_lo_ymm, row26_lo_ymm, row37_lo_ymm;
    __m256 row04_hi_ymm, row15_hi_ymm, row26_hi_ymm, row37_hi_ymm;
    row04_lo_ymm = shuffle(col01_lo_ymm, col23_lo_ymm, _MM_SHUFFLE(1, 0, 1, 0)); // [c0[0],c1[0],c2[0],c3[0], c0[4],c1[4],c2[4],c3[4]]
    row15_lo_ymm = shuffle(col01_lo_ymm, col23_lo_ymm, _MM_SHUFFLE(3, 2, 3, 2)); // [c0[1],c1[1],c2[1],c3[1], c0[5],c1[5],c2[5],c3[5]]
    row26_lo_ymm = shuffle(col01_hi_ymm, col23_hi_ymm, _MM_SHUFFLE(1, 0, 1, 0));
    row37_lo_ymm = shuffle(col01_hi_ymm, col23_hi_ymm, _MM_SHUFFLE(3, 2, 3, 2));
    row04_hi_ymm = shuffle(col45_lo_ymm, col67_lo_ymm, _MM_SHUFFLE(1, 0, 1, 0));
    row15_hi_ymm = shuffle(col45_lo_ymm, col67_lo_ymm, _MM_SHUFFLE(3, 2, 3, 2));
    row26_hi_ymm = shuffle(col45_hi_ymm, col67_hi_ymm, _MM_SHUFFLE(1, 0, 1, 0));
    row37_hi_ymm = shuffle(col45_hi_ymm, col67_hi_ymm, _MM_SHUFFLE(3, 2, 3, 2));

    // permute interleave 4-element groups from different vectors
    // c_rowX_ymm variable stores the correct transposed row data
    __m256 c_row0_ymm = permute2f128(row04_lo_ymm, row04_hi_ymm, 0x20); // [c0[0]..c3[0], c4[0]..c7[0]] -> row 0
    __m256 c_row1_ymm = permute2f128(row15_lo_ymm, row15_hi_ymm, 0x20); // [c0[1]..c3[1], c4[1]..c7[1]] -> row 1
    __m256 c_row2_ymm = permute2f128(row26_lo_ymm, row26_hi_ymm, 0x20); // [c0[2]..c3[2], c4[2]..c7[2]] -> row 2
    __m256 c_row3_ymm = permute2f128(row37_lo_ymm, row37_hi_ymm, 0x20); // [c0[3]..c3[3], c4[3]..c7[3]] -> row 3
    __m256 c_row4_ymm = permute2f128(row04_lo_ymm, row04_hi_ymm, 0x31); // [c0[4]..c3[4], c4[4]..c7[4]] -> row 4
    __m256 c_row5_ymm = permute2f128(row15_lo_ymm, row15_hi_ymm, 0x31); // [c0[5]..c3[5], c4[5]..c7[5]] -> row 5
    __m256 c_row6_ymm = permute2f128(row26_lo_ymm, row26_hi_ymm, 0x31); // [c0[6]..c3[6], c4[6]..c7[6]] -> row 6
    __m256 c_row7_ymm = permute2f128(row37_lo_ymm, row37_hi_ymm, 0x31); // [c0[7]..c3[7], c4[7]..c7[7]] -> row 7

    // write back result：C = alpha * (A*B) + beta * C
    __m256 alpha_ymm = _mm256_set1_ps(alpha);
    __m256 beta_ymm = _mm256_set1_ps(beta);

    float *c_ptr = c; 
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row0_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row1_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row2_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row3_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row4_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row5_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row6_ymm))); c_ptr += ldc;
    store(c_ptr, add(mul(beta_ymm, load<__m256>(c_ptr)), mul(alpha_ymm, c_row7_ymm)));

}

template <int64_t RM, int64_t RN>
void AddDot_8x8_kernel_double(
    int64_t k, double *a, double *b, double *c, int64_t ldc, MicroKernelCtxType<double> *ctx) {
    int64_t p;
    double alpha = 1.0, beta = 1.0;

    // 16 accumulators for 8x8 double matrix (8 columns * 2 regs/col)
    __m256d c_col0_ymm0 = setzeros<__m256d>(); __m256d c_col0_ymm1 = setzeros<__m256d>();
    __m256d c_col1_ymm0 = setzeros<__m256d>(); __m256d c_col1_ymm1 = setzeros<__m256d>();
    __m256d c_col2_ymm0 = setzeros<__m256d>(); __m256d c_col2_ymm1 = setzeros<__m256d>();
    __m256d c_col3_ymm0 = setzeros<__m256d>(); __m256d c_col3_ymm1 = setzeros<__m256d>();
    __m256d c_col4_ymm0 = setzeros<__m256d>(); __m256d c_col4_ymm1 = setzeros<__m256d>();
    __m256d c_col5_ymm0 = setzeros<__m256d>(); __m256d c_col5_ymm1 = setzeros<__m256d>();
    __m256d c_col6_ymm0 = setzeros<__m256d>(); __m256d c_col6_ymm1 = setzeros<__m256d>();
    __m256d c_col7_ymm0 = setzeros<__m256d>(); __m256d c_col7_ymm1 = setzeros<__m256d>();
    
    // define a0 a1 b0 b1, initializ a0 and b0
    __m256d tmp_ymm;
    __m256d a_col_ymm0, b_row_ymm0;
    __m256d a_col_ymm1, b_row_ymm1;

    for (p = 0; p < k; ++p) {
        a_col_ymm0 = load<__m256d>(a); 
        a_col_ymm1 = load<__m256d>(a + 4);
        b_row_ymm0 = load<__m256d>(b); 
        b_row_ymm1 = load<__m256d>(b + 4);

        // handle a[0:3] * b[0] and a[4:7] * b[0] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm0, 0x00);
        c_col0_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col0_ymm0);
        c_col0_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col0_ymm1);

        // handle a[0:3] * b[1] and a[4:7] * b[1] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm0, 0x55);
        c_col1_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col1_ymm0);
        c_col1_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col1_ymm1);

         // handle a[0:3] * b[2] and a[4:7] * b[2] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm0, 0xAA);
        c_col2_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col2_ymm0);
        c_col2_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col2_ymm1);

        // handle a[0:3] * b[3] and a[4:7] * b[3] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm0, 0xFF);
        c_col3_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col3_ymm0);
        c_col3_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col3_ymm1);

        // handle a[0:3] * b[4] and a[4:7] * b[4] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm1, 0x00);
        c_col4_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col4_ymm0);
        c_col4_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col4_ymm1);

        // handle a[0:3] * b[5] and a[4:7] * b[5] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm1, 0x55);
        c_col5_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col5_ymm0);
        c_col5_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col5_ymm1);

        // handle a[0:3] * b[6] and a[4:7] * b[6] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm1, 0xAA);
        c_col6_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col6_ymm0);
        c_col6_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col6_ymm1);

        // handle a[0:3] * b[4] and a[4:7] * b[4] for 1st iteration
        tmp_ymm = permute4x64(b_row_ymm1, 0xFF);
        c_col7_ymm0 = madd(a_col_ymm0, tmp_ymm, c_col7_ymm0);
        c_col7_ymm1 = madd(a_col_ymm1, tmp_ymm, c_col7_ymm1);

        // update pointer
        a += 8;
        b += 8;
    }

    // Correct 8x8 double matrix transposition
    // Step 1: Unpack 4x4 blocks
    __m256d col01_lo_ymm0, col01_hi_ymm0, col23_lo_ymm0, col23_hi_ymm0;
    __m256d col01_lo_ymm1, col01_hi_ymm1, col23_lo_ymm1, col23_hi_ymm1;
    col01_lo_ymm0 = unpacklo(c_col0_ymm0, c_col1_ymm0);
    col01_hi_ymm0 = unpackhi(c_col0_ymm0, c_col1_ymm0);
    col23_lo_ymm0 = unpacklo(c_col2_ymm0, c_col3_ymm0);
    col23_hi_ymm0 = unpackhi(c_col2_ymm0, c_col3_ymm0);
    col01_lo_ymm1 = unpacklo(c_col0_ymm1, c_col1_ymm1);
    col01_hi_ymm1 = unpackhi(c_col0_ymm1, c_col1_ymm1);
    col23_lo_ymm1 = unpacklo(c_col2_ymm1, c_col3_ymm1);
    col23_hi_ymm1 = unpackhi(c_col2_ymm1, c_col3_ymm1);

    // Step 2: Permute to get rows 0-3 (first 4 elements)
    __m256d row0_ymm0, row1_ymm0, row2_ymm0, row3_ymm0;
    row0_ymm0 = permute2f128(col01_lo_ymm0, col23_lo_ymm0, 0x20);
    row1_ymm0 = permute2f128(col01_hi_ymm0, col23_hi_ymm0, 0x20);
    row2_ymm0 = permute2f128(col01_lo_ymm0, col23_lo_ymm0, 0x31);
    row3_ymm0 = permute2f128(col01_hi_ymm0, col23_hi_ymm0, 0x31);

    // Step 3: Repeat for other half
    col01_lo_ymm0 = unpacklo(c_col4_ymm0, c_col5_ymm0);
    col01_hi_ymm0 = unpackhi(c_col4_ymm0, c_col5_ymm0);
    col23_lo_ymm0 = unpacklo(c_col6_ymm0, c_col7_ymm0);
    col23_hi_ymm0 = unpackhi(c_col6_ymm0, c_col7_ymm0);

    // Step 4: Permute to get rows 0-3 (last 4 elements)
    __m256d row0_ymm1, row1_ymm1, row2_ymm1, row3_ymm1;
    row0_ymm1 = permute2f128(col01_lo_ymm0, col23_lo_ymm0, 0x20);
    row1_ymm1 = permute2f128(col01_hi_ymm0, col23_hi_ymm0, 0x20);
    row2_ymm1 = permute2f128(col01_lo_ymm0, col23_lo_ymm0, 0x31);
    row3_ymm1 = permute2f128(col01_hi_ymm0, col23_hi_ymm0, 0x31);

    // Step 5: Repeat for rows 4-7
    col01_lo_ymm0 = unpacklo(c_col4_ymm1, c_col5_ymm1);
    col01_hi_ymm0 = unpackhi(c_col4_ymm1, c_col5_ymm1);
    col23_lo_ymm0 = unpacklo(c_col6_ymm1, c_col7_ymm1);
    col23_hi_ymm0 = unpackhi(c_col6_ymm1, c_col7_ymm1);

    __m256d row4_ymm0, row5_ymm0, row6_ymm0, row7_ymm0;
    row4_ymm0 = permute2f128(col01_lo_ymm1, col23_lo_ymm1, 0x20);
    row5_ymm0 = permute2f128(col01_hi_ymm1, col23_hi_ymm1, 0x20);
    row6_ymm0 = permute2f128(col01_lo_ymm1, col23_lo_ymm1, 0x31);
    row7_ymm0 = permute2f128(col01_hi_ymm1, col23_hi_ymm1, 0x31);

    __m256d row4_ymm1, row5_ymm1, row6_ymm1, row7_ymm1;
    row4_ymm1 = permute2f128(col01_lo_ymm0, col23_lo_ymm0, 0x20);
    row5_ymm1 = permute2f128(col01_hi_ymm0, col23_hi_ymm0, 0x20);
    row6_ymm1 = permute2f128(col01_lo_ymm0, col23_lo_ymm0, 0x31);
    row7_ymm1 = permute2f128(col01_hi_ymm0, col23_hi_ymm0, 0x31);

    // write back result: C = alpha * (A*B) + beta * C
    __m256d alpha_ymm = set1<__m256d>(alpha);
    __m256d beta_ymm = set1<__m256d>(beta);

    // add the transposed result to matrix C
    double *c_ptr = c;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row0_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row0_ymm1)));
    c_ptr += ldc;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row1_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row1_ymm1)));
    c_ptr += ldc;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row2_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row2_ymm1)));
    c_ptr += ldc;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row3_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row3_ymm1)));
    c_ptr += ldc;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row4_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row4_ymm1)));
    c_ptr += ldc;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row5_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row5_ymm1)));
    c_ptr += ldc;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row6_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row6_ymm1)));
    c_ptr += ldc;
    store(c_ptr,     add(mul(beta_ymm, load<__m256d>(c_ptr)),     mul(alpha_ymm, row7_ymm0)));
    store(c_ptr + 4, add(mul(beta_ymm, load<__m256d>(c_ptr + 4)), mul(alpha_ymm, row7_ymm1)));
}

template <typename TA, typename TB, typename TC, int64_t RM, int64_t RN>
void AddDot_8x8_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc, MicroKernelCtxType<TB> *ctx) {
    if constexpr (std::is_same_v<TA, float> && std::is_same_v<TB, float> &&
                  std::is_same_v<TC, float>) {
        AddDot_8x8_kernel_float<RM, RN>(k, a, b, c, ldc, ctx);
    } else if constexpr (std::is_same_v<TA, double> && std::is_same_v<TB, double> &&
                         std::is_same_v<TC, double>) {
        AddDot_8x8_kernel_double<RM, RN>(k, a, b, c, ldc, ctx);
    }
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // X86_GEMM_8x8_KERNEL_HPP_