#ifndef X86_GEMM_4X4_KERNEL_HPP_
#define X86_GEMM_4X4_KERNEL_HPP_

#include <arch/gemm_kernels.hpp>

namespace gemm {
namespace detail {

template <int64_t RM = 4, int64_t RN = 4>
void AddDot_4x4_kernel_float(int64_t k, float *a, float *b, float *c, int64_t ldc) {
    int i;
    float alpha = 1.0f, beta = 1.0f;
    float *c0_3_0, *c0_3_1, *c0_3_2, *c0_3_3;
    
    // define matrix a0_3 and c0_3
    __m128 a0_3_xmm;
    __m128 c0_3_0_xmm, c0_3_1_xmm, c0_3_2_xmm, c0_3_3_xmm;

    // accumulator
    __m128 a0_3b_0_xmm, a0_3b_1_xmm, a0_3b_2_xmm, a0_3b_3_xmm;
    a0_3b_0_xmm = _mm_setzero_ps();
    a0_3b_1_xmm = _mm_setzero_ps();
    a0_3b_2_xmm = _mm_setzero_ps();
    a0_3b_3_xmm = _mm_setzero_ps();

    // pre-fetch a, b, and c data
    __asm__ volatile( "prefetcht0 0(%0)          \n\t" : :"r"(a)  );
	__asm__ volatile( "prefetcht2 0(%0)          \n\t" : :"r"(b)  );
	__asm__ volatile( "prefetcht0 0(%0)          \n\t" : :"r"(c)  );

    int64_t k_align = k / 4;
    int64_t k_left = k % 4;
    for (i = 0; i < k_align; ++i) {
        // compute 1st k value
        a0_3_xmm = _mm_load_ps(a);
        a0_3b_0_xmm = _mm_add_ps(a0_3b_0_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[0])));
        a0_3b_1_xmm = _mm_add_ps(a0_3b_1_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[1])));
        a0_3b_2_xmm = _mm_add_ps(a0_3b_2_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[2])));
        a0_3b_3_xmm = _mm_add_ps(a0_3b_3_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[3])));

        // compute 2ed k value
        a0_3_xmm = _mm_load_ps(a + 4);
        a0_3b_0_xmm = _mm_add_ps(a0_3b_0_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[4])));
        a0_3b_1_xmm = _mm_add_ps(a0_3b_1_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[5])));
        a0_3b_2_xmm = _mm_add_ps(a0_3b_2_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[6])));
        a0_3b_3_xmm = _mm_add_ps(a0_3b_3_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[7])));

        __asm__ volatile( "prefetcht0 64(%0)          \n\t" : :"r"(a + 16)  );
        // compute 3rd k value
        a0_3_xmm = _mm_load_ps(a + 8);
        a0_3b_0_xmm = _mm_add_ps(a0_3b_0_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[8])));
        a0_3b_1_xmm = _mm_add_ps(a0_3b_1_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[9])));
        a0_3b_2_xmm = _mm_add_ps(a0_3b_2_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[10])));
        a0_3b_3_xmm = _mm_add_ps(a0_3b_3_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[11])));

        // compute 4th k value
        a0_3_xmm = _mm_load_ps(a + 12);
        a0_3b_0_xmm = _mm_add_ps(a0_3b_0_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[12])));
        a0_3b_1_xmm = _mm_add_ps(a0_3b_1_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[13])));
        a0_3b_2_xmm = _mm_add_ps(a0_3b_2_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[14])));
        a0_3b_3_xmm = _mm_add_ps(a0_3b_3_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[15])));

        // increment
        a += 16;
        b += 16;
    } 

    for (i = 0; i < k_left; ++i) {
        a0_3_xmm = _mm_load_ps(a);
        a0_3b_0_xmm = _mm_add_ps(a0_3b_0_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[0])));
        a0_3b_1_xmm = _mm_add_ps(a0_3b_1_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[1])));
        a0_3b_2_xmm = _mm_add_ps(a0_3b_2_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[2])));
        a0_3b_3_xmm = _mm_add_ps(a0_3b_3_xmm, _mm_mul_ps(a0_3_xmm, _mm_set1_ps(b[3])));
        a += 4;
        b += 4;
    }

    __m128 alpha_xmm = _mm_set1_ps(alpha);
    __m128 beta_xmm = _mm_set1_ps(beta);

    // 1st column
    c0_3_0 = c + 0 * ldc;
    c0_3_0_xmm = _mm_load_ps(c0_3_0);
    c0_3_0_xmm = _mm_add_ps(_mm_mul_ps(beta_xmm, c0_3_0_xmm), _mm_mul_ps(alpha_xmm, a0_3b_0_xmm));
    _mm_store_ps(c0_3_0, c0_3_0_xmm);

    // 2ed col
    c0_3_1 = c + 1 * ldc;
    c0_3_1_xmm = _mm_load_ps(c0_3_1);
    c0_3_1_xmm = _mm_add_ps(_mm_mul_ps(beta_xmm, c0_3_1_xmm), _mm_mul_ps(alpha_xmm, a0_3b_1_xmm));
    _mm_store_ps(c0_3_1, c0_3_1_xmm);

    // 3rd col
    c0_3_2 = c + 2 * ldc;
    c0_3_2_xmm = _mm_load_ps(c0_3_2);
    c0_3_2_xmm = _mm_add_ps(_mm_mul_ps(beta_xmm, c0_3_2_xmm), _mm_mul_ps(alpha_xmm, a0_3b_2_xmm));
    _mm_store_ps(c0_3_2, c0_3_2_xmm);

    // 4th col
    c0_3_3 = c + 3 * ldc;
    c0_3_3_xmm = _mm_load_ps(c0_3_3);
    c0_3_3_xmm = _mm_add_ps(_mm_mul_ps(beta_xmm, c0_3_3_xmm), _mm_mul_ps(alpha_xmm, a0_3b_3_xmm));
    _mm_store_ps(c0_3_3, c0_3_3_xmm);
}


template <int64_t RM = 4, int64_t RN = 4>
void AddDot_4x4_kernel_double(int64_t k, double *a, double *b, double *c, int64_t ldc) {
    double c_00_reg, c_01_reg, c_02_reg, c_03_reg;
    double c_10_reg, c_11_reg, c_12_reg, c_13_reg;
    double c_20_reg, c_21_reg, c_22_reg, c_23_reg;
    double c_30_reg, c_31_reg, c_32_reg, c_33_reg;

    c_00_reg = 0.0; c_01_reg = 0.0; c_02_reg = 0.0; c_03_reg = 0.0;
    c_10_reg = 0.0; c_11_reg = 0.0; c_12_reg = 0.0; c_13_reg = 0.0;
    c_20_reg = 0.0; c_21_reg = 0.0; c_22_reg = 0.0; c_23_reg = 0.0;
    c_30_reg = 0.0; c_31_reg = 0.0; c_32_reg = 0.0; c_33_reg = 0.0;

    int p;
    for (p = 0; p < k; p++) {
        // pre-load A-row to register A(i, p) = A[j * RM + i]
        double a_0p_reg = a[p * RM + 0];
        double a_1p_reg = a[p * RM + 1];
        double a_2p_reg = a[p * RM + 2];
        double a_3p_reg = a[p * RM + 3];

        // pre-load B-column to register B(p, j) = B[j * RN + i]
        double b_p0_reg = b[p * RN + 0];
        double b_p1_reg = b[p * RN + 1]; 
        double b_p2_reg = b[p * RN + 2];
        double b_p3_reg = b[p * RN + 3];

        // 1st row
        c_00_reg += a_0p_reg * b_p0_reg;
        c_01_reg += a_0p_reg * b_p1_reg;
        c_02_reg += a_0p_reg * b_p2_reg;
        c_03_reg += a_0p_reg * b_p3_reg;

        // 2ed row
        c_10_reg += a_1p_reg * b_p0_reg;
        c_11_reg += a_1p_reg * b_p1_reg;
        c_12_reg += a_1p_reg * b_p2_reg;
        c_13_reg += a_1p_reg * b_p3_reg;
        
        // 3rd row
        c_20_reg += a_2p_reg * b_p0_reg;
        c_21_reg += a_2p_reg * b_p1_reg;
        c_22_reg += a_2p_reg * b_p2_reg;
        c_23_reg += a_2p_reg * b_p3_reg;
        
        // 4th row
        c_30_reg += a_3p_reg * b_p0_reg;
        c_31_reg += a_3p_reg * b_p1_reg;
        c_32_reg += a_3p_reg * b_p2_reg;
        c_33_reg += a_3p_reg * b_p3_reg;
    }

    // C(i, j) = C[j * ldc + i]
    c[0 * ldc + 0] += c_00_reg; c[1 * ldc + 0] += c_01_reg; c[2 * ldc + 0] += c_02_reg; c[3 * ldc + 0] += c_03_reg;
    c[0 * ldc + 1] += c_10_reg; c[1 * ldc + 1] += c_11_reg; c[2 * ldc + 1] += c_12_reg; c[3 * ldc + 1] += c_13_reg;
    c[0 * ldc + 2] += c_20_reg; c[1 * ldc + 2] += c_21_reg; c[2 * ldc + 2] += c_22_reg; c[3 * ldc + 2] += c_23_reg;
    c[0 * ldc + 3] += c_30_reg; c[1 * ldc + 3] += c_31_reg; c[2 * ldc + 3] += c_32_reg; c[3 * ldc + 3] += c_33_reg;
};

template <typename TA, typename TB, typename TC, int64_t RM = 4, int64_t RN = 4>
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

}
}
#endif // X86_GEMM_4X4_KERNEL_HPP_