#ifndef TINYBLAS_COMMON_HPP_
#define TINYBLAS_COMMON_HPP_

#include <cstdint>

#define MEMORY_ALIGNMENT 32
#define QK8_0 32
#define QK4_0 32

namespace tinyBLAS {

// The following code is adapted from 
// https://github.com/ggml-org/ggml/blob/master/src/ggml-cpu/arch/x86/repack.h

template <int K> 
constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> 
struct Block_QK_0 {
    float d[N];                         // deltas for N qK_0 blocks
    int8_t qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

using Block_Q8_0x4Type = Block_QK_0<8, 4>;
using Block_Q8_0x8Type = Block_QK_0<8, 8>;

}

#endif // TINYBLAS_COMMON_HPP_