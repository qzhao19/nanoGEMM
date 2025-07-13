#include <arch/x86/gemm_4x4_kernel.hpp>

namespace gemm {
namespace detail {

template <int64_t RM = 4, int64_t RN = 4>
inline void AddDot_4x4_kernel_float(int64_t k, float *a, float *b, float *c, int64_t ldc) {
    float c_00_reg, c_01_reg, c_02_reg, c_03_reg;
    float c_10_reg, c_11_reg, c_12_reg, c_13_reg;
    float c_20_reg, c_21_reg, c_22_reg, c_23_reg;
    float c_30_reg, c_31_reg, c_32_reg, c_33_reg;

    c_00_reg = 0.0; c_01_reg = 0.0; c_02_reg = 0.0; c_03_reg = 0.0;
    c_10_reg = 0.0; c_11_reg = 0.0; c_12_reg = 0.0; c_13_reg = 0.0;
    c_20_reg = 0.0; c_21_reg = 0.0; c_22_reg = 0.0; c_23_reg = 0.0;
    c_30_reg = 0.0; c_31_reg = 0.0; c_32_reg = 0.0; c_33_reg = 0.0;

    int p;
    for (p = 0; p < k; p++) {
        // pre-load A-row to register A(i, p) = A[j * RM + i]
        float a_0p_reg = a[p * RM + 0];
        float a_1p_reg = a[p * RM + 1];
        float a_2p_reg = a[p * RM + 2];
        float a_3p_reg = a[p * RM + 3];

        // pre-load B-column to register B(p, j) = B[j * RN + i]
        float b_p0_reg = b[0 * RN + p];
        float b_p1_reg = b[1 * RN + p]; 
        float b_p2_reg = b[2 * RN + p];
        float b_p3_reg = b[3 * RN + p];

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
inline void AddDot_4x4_kernel_double(int64_t k, double *a, double *b, double *c, int64_t ldc) {
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
        double b_p0_reg = b[0 * RN + p];
        double b_p1_reg = b[1 * RN + p]; 
        double b_p2_reg = b[2 * RN + p];
        double b_p3_reg = b[3 * RN + p];

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
inline void AddDot_4x4_kernel(int64_t k, TA *a, TB *b, TC *c, int64_t ldc) {
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



