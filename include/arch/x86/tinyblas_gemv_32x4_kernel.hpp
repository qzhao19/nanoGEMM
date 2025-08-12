#ifndef X86_TINYBLAS_GEMV_32X4_KERNEL_HPP_
#define X86_TINYBLAS_GEMV_32X4_KERNEL_HPP_

#include <arch/tinyblas_kernels.hpp>

namespace tinyBLAS {
namespace detail {

template <int64_t RM, int64_t RN>
void AddDot_32x4_kernel_float(int64_t k, float *a, float *x, float *y) {
    int64_t i;
    float alpha = 1.0f, beta = 1.0f;

    // 
    float *y0_7, *y8_15, *y16_23, *y24_31;

    __m256 a0_7_ymm, a8_15_ymm, a16_23_ymm, a24_31_ymm;
    __m256 tmp_ymm;

    // accumulator
    __m256 a0_7x_0_ymm = setzeros<__m256>();
    __m256 a0_7x_1_ymm = setzeros<__m256>();
    __m256 a0_7x_2_ymm = setzeros<__m256>();
    __m256 a0_7x_3_ymm = setzeros<__m256>();
    
    // Second row
    __m256 a8_15x_0_ymm = setzeros<__m256>();
    __m256 a8_15x_1_ymm = setzeros<__m256>();
    __m256 a8_15x_2_ymm = setzeros<__m256>();
    __m256 a8_15x_3_ymm = setzeros<__m256>();
    
    // Third row
    __m256 a16_23x_0_ymm = setzeros<__m256>();
    __m256 a16_23x_1_ymm = setzeros<__m256>();
    __m256 a16_23x_2_ymm = setzeros<__m256>();
    __m256 a16_23x_3_ymm = setzeros<__m256>();
    
    // Fourth row
    __m256 a24_31x_0_ymm = setzeros<__m256>();
    __m256 a24_31x_1_ymm = setzeros<__m256>();
    __m256 a24_31x_2_ymm = setzeros<__m256>();
    __m256 a24_31x_3_ymm = setzeros<__m256>();

    // prefetch initial data
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(a));
    __asm__ volatile("prefetcht2 0(%0)          \n\t" : : "r"(x));
    __asm__ volatile("prefetcht0 0(%0)          \n\t" : : "r"(y));

    int64_t k_blocks = k / 2;
    int64_t k_remainder = k % 2;

    for (i = 0; i < k_blocks; ++i) {
        // 1st iteration
        a0_7_ymm = load<__m256>(a);
        a8_15_ymm = load<__m256>(a + 8);
        a16_23_ymm = load<__m256>(a + 16);
        a24_31_ymm = load<__m256>(a + 24);

        // handle x[0]
        tmp_ymm = set1<__m256>(x[0]);
        a0_7x_0_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_0_ymm);
        a8_15x_0_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_0_ymm);
        a16_23x_0_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_0_ymm);
        a24_31x_0_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_0_ymm);
        
        // handle x[1]
        tmp_ymm = set1<__m256>(x[1]);
        a0_7x_1_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_1_ymm);
        a8_15x_1_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_1_ymm);
        a16_23x_1_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_1_ymm);
        a24_31x_1_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_1_ymm);
        
        // handle x[2]
        tmp_ymm = set1<__m256>(x[2]);
        a0_7x_2_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_2_ymm);
        a8_15x_2_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_2_ymm);
        a16_23x_2_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_2_ymm);
        a24_31x_2_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_2_ymm);
        
        // handle x[3]
        tmp_ymm = set1<__m256>(x[3]);
        a0_7x_3_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_3_ymm);
        a8_15x_3_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_3_ymm);
        a16_23x_3_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_3_ymm);
        a24_31x_3_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_3_ymm);

        // move to next data
        a += 32;
        x += 4;
        
        // 2ed iteration
        a0_7_ymm = load<__m256>(a);
        a8_15_ymm = load<__m256>(a + 8);
        a16_23_ymm = load<__m256>(a + 16);
        a24_31_ymm = load<__m256>(a + 24);

        // handle x[0]
        tmp_ymm = set1<__m256>(x[0]);
        a0_7x_0_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_0_ymm);
        a8_15x_0_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_0_ymm);
        a16_23x_0_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_0_ymm);
        a24_31x_0_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_0_ymm);
        
        // handle x[1] - x[3]
        tmp_ymm = set1<__m256>(x[1]);
        a0_7x_1_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_1_ymm);
        a8_15x_1_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_1_ymm);
        a16_23x_1_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_1_ymm);
        a24_31x_1_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_1_ymm);
        
        tmp_ymm = set1<__m256>(x[2]);
        a0_7x_2_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_2_ymm);
        a8_15x_2_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_2_ymm);
        a16_23x_2_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_2_ymm);
        a24_31x_2_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_2_ymm);
        
        tmp_ymm = set1<__m256>(x[3]);
        a0_7x_3_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_3_ymm);
        a8_15x_3_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_3_ymm);
        a16_23x_3_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_3_ymm);
        a24_31x_3_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_3_ymm);
        
        a += 32;
        x += 4;
    }

    // handle remaining
    if (k_remainder > 0) {
        a0_7_ymm = load<__m256>(a);
        a8_15_ymm = load<__m256>(a + 8);
        a16_23_ymm = load<__m256>(a + 16);
        a24_31_ymm = load<__m256>(a + 24);

        for (i = 0; i < k_remainder; ++i) {
            tmp_ymm = set1<__m256>(x[i]);
            
            if (i == 0) {
                a0_7x_0_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_0_ymm);
                a8_15x_0_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_0_ymm);
                a16_23x_0_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_0_ymm);
                a24_31x_0_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_0_ymm);
            } else if (i == 1) {
                a0_7x_1_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_1_ymm);
                a8_15x_1_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_1_ymm);
                a16_23x_1_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_1_ymm);
                a24_31x_1_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_1_ymm);
            } else if (i == 2) {
                a0_7x_2_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_2_ymm);
                a8_15x_2_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_2_ymm);
                a16_23x_2_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_2_ymm);
                a24_31x_2_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_2_ymm);
            } else if (i == 3) {
                a0_7x_3_ymm = madd<__m256>(a0_7_ymm, tmp_ymm, a0_7x_3_ymm);
                a8_15x_3_ymm = madd<__m256>(a8_15_ymm, tmp_ymm, a8_15x_3_ymm);
                a16_23x_3_ymm = madd<__m256>(a16_23_ymm, tmp_ymm, a16_23x_3_ymm);
                a24_31x_3_ymm = madd<__m256>(a24_31_ymm, tmp_ymm, a24_31x_3_ymm);
            }
        }
    }

    // store results
    y0_7 = y;
    y8_15 = y + 8;
    y16_23 = y + 16;
    y24_31 = y + 24;
    
    tmp_ymm = load<__m256>(y0_7);
    tmp_ymm = add(tmp_ymm, a0_7x_0_ymm);
    store(y0_7, tmp_ymm);
    
    tmp_ymm = load<__m256>(y8_15);
    tmp_ymm = add(tmp_ymm, a8_15x_0_ymm);
    store(y8_15, tmp_ymm);
    
    tmp_ymm = load<__m256>(y16_23);
    tmp_ymm = add(tmp_ymm, a16_23x_0_ymm);
    store(y16_23, tmp_ymm);
    
    tmp_ymm = load<__m256>(y24_31);
    tmp_ymm = add(tmp_ymm, a24_31x_0_ymm);
    store(y24_31, tmp_ymm);
    
    y0_7 = y + 1;
    y8_15 = y + 1 + 8;
    y16_23 = y + 1 + 16;
    y24_31 = y + 1 + 24;
    
    tmp_ymm = load<__m256>(y0_7);
    tmp_ymm = add(tmp_ymm, a0_7x_1_ymm);
    store(y0_7, tmp_ymm);
    
    tmp_ymm = load<__m256>(y8_15);
    tmp_ymm = add(tmp_ymm, a8_15x_1_ymm);
    store(y8_15, tmp_ymm);
    
    tmp_ymm = load<__m256>(y16_23);
    tmp_ymm = add(tmp_ymm, a16_23x_1_ymm);
    store(y16_23, tmp_ymm);
    
    tmp_ymm = load<__m256>(y24_31);
    tmp_ymm = add(tmp_ymm, a24_31x_1_ymm);
    store(y24_31, tmp_ymm);
    
    y0_7 = y + 2;
    y8_15 = y + 2 + 8;
    y16_23 = y + 2 + 16;
    y24_31 = y + 2 + 24;
    
    tmp_ymm = load<__m256>(y0_7);
    tmp_ymm = add(tmp_ymm, a0_7x_2_ymm);
    store(y0_7, tmp_ymm);
    
    tmp_ymm = load<__m256>(y8_15);
    tmp_ymm = add(tmp_ymm, a8_15x_2_ymm);
    store(y8_15, tmp_ymm);
    
    tmp_ymm = load<__m256>(y16_23);
    tmp_ymm = add(tmp_ymm, a16_23x_2_ymm);
    store(y16_23, tmp_ymm);
    
    tmp_ymm = load<__m256>(y24_31);
    tmp_ymm = add(tmp_ymm, a24_31x_2_ymm);
    store(y24_31, tmp_ymm);
    
    y0_7 = y + 3;
    y8_15 = y + 3 + 8;
    y16_23 = y + 3 + 16;
    y24_31 = y + 3 + 24;
    
    tmp_ymm = load<__m256>(y0_7);
    tmp_ymm = add(tmp_ymm, a0_7x_3_ymm);
    store(y0_7, tmp_ymm);
    
    tmp_ymm = load<__m256>(y8_15);
    tmp_ymm = add(tmp_ymm, a8_15x_3_ymm);
    store(y8_15, tmp_ymm);
    
    tmp_ymm = load<__m256>(y16_23);
    tmp_ymm = add(tmp_ymm, a16_23x_3_ymm);
    store(y16_23, tmp_ymm);
    
    tmp_ymm = load<__m256>(y24_31);
    tmp_ymm = add(tmp_ymm, a24_31x_3_ymm);
    store(y24_31, tmp_ymm);
};

template <typename TA, typename TX, typename TY, int64_t RM, int64_t RN>
void AddDot_32x4_kernel(int64_t k, TA *a, TX *x, TY *y) {
    if constexpr (std::is_same_v<TA, float> && std::is_same_v<TX, float> &&
                  std::is_same_v<TY, float>) {
        AddDot_32x4_kernel_float<RM, RN>(k, a, x, y);
    } 
    // else if constexpr (std::is_same_v<TA, double> && std::is_same_v<TB, double> &&
    //                      std::is_same_v<TC, double>) {
    //     AddDot_4x4_kernel_double<RM, RN>(k, a, b, c, ctx);
    // }
};

} // detail
} // tinyBLAS
#endif // X86_TINYBLAS_GEMV_32X4_KERNEL_HPP_