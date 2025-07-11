#include <arch/x86/gemm_4x4_kernel.hpp>

namespace gemm {
namespace detail {

template <int64_t RM = 4, int64_t RN = 4>
inline void AddDot_4x4_kernel(int64_t k, float *a, float *b, float *c, int64_t ldc) {
    register float c_00_reg, c_01_reg, c_02_reg, c_03_reg;
    register float c_10_reg, c_11_reg, c_12_reg, c_13_reg;
    register float c_20_reg, c_21_reg, c_22_reg, c_23_reg;
    register float c_30_reg, c_31_reg, c_32_reg, c_33_reg;

    c_00_reg = 0.0; c_01_reg = 0.0; c_02_reg = 0.0; c_03_reg = 0.0;
    c_10_reg = 0.0; c_11_reg = 0.0; c_12_reg = 0.0; c_13_reg = 0.0;
    c_20_reg = 0.0; c_21_reg = 0.0; c_22_reg = 0.0; c_23_reg = 0.0;
    c_30_reg = 0.0; c_31_reg = 0.0; c_32_reg = 0.0; c_33_reg = 0.0;

    int p;
    for (p = 0; p < k; p++) {
        // pre-load A-row to register A(i, p) = A[j * RM + i]
        register float a_0p_reg = a[p * RM + 0];
        register float a_1p_reg = a[p * RM + 1];
        register float a_2p_reg = a[p * RM + 2];
        register float a_3p_reg = a[p * RM + 3];

        // pre-load B-column to register B(p, j) = B[j * RN + i]
        register float b_p0_reg = b[0 * RN + p];
        register float b_p1_reg = b[1 * RN + p]; 
        register float b_p2_reg = b[2 * RN + p];
        register float b_p3_reg = b[3 * RN + p];

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

template <int64_t RM = 4, int64_t RN = 4>
inline void AddDot_4x4_kernel(int64_t k, double *a, double *b, double *c, int64_t ldc) {
    register double c_00_reg, c_01_reg, c_02_reg, c_03_reg;
    register double c_10_reg, c_11_reg, c_12_reg, c_13_reg;
    register double c_20_reg, c_21_reg, c_22_reg, c_23_reg;
    register double c_30_reg, c_31_reg, c_32_reg, c_33_reg;

    c_00_reg = 0.0; c_01_reg = 0.0; c_02_reg = 0.0; c_03_reg = 0.0;
    c_10_reg = 0.0; c_11_reg = 0.0; c_12_reg = 0.0; c_13_reg = 0.0;
    c_20_reg = 0.0; c_21_reg = 0.0; c_22_reg = 0.0; c_23_reg = 0.0;
    c_30_reg = 0.0; c_31_reg = 0.0; c_32_reg = 0.0; c_33_reg = 0.0;

    int p;
    for (p = 0; p < k; p++) {
        // pre-load A-row to register A(i, p) = A[j * RM + i]
        register double a_0p_reg = a[p * RM + 0];
        register double a_1p_reg = a[p * RM + 1];
        register double a_2p_reg = a[p * RM + 2];
        register double a_3p_reg = a[p * RM + 3];

        // pre-load B-column to register B(p, j) = B[j * RN + i]
        register double b_p0_reg = b[0 * RN + p];
        register double b_p1_reg = b[1 * RN + p]; 
        register double b_p2_reg = b[2 * RN + p];
        register double b_p3_reg = b[3 * RN + p];

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

}
}



