#ifndef X86_KERNELS_HPP_
#define X86_KERNELS_HPP_

#include <cmath>
#include <cstring>
#include <new>

namespace gemm {
namespace detail {

#if defined(__SSE__)
#include <smmintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h> 
#endif

// 
#if defined(__SSE__)
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE4_2__

#if defined(__AVX2__)
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif  // __AVX2__

/**
 * Computes a * b + c.
 * apply _mm256_fmadd_ps if enable fma
 */
template<typename T>
inline T madd(T a, T b, T c) {
    return add(mul(a, b), c);
}

#if defined(__SSE__) || defined(__AVX2__)
#if defined(__FMA__)
template<>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#endif

#if defined(__SSE__)
inline float hsum(__m128 x) {
    __m128 t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX2__)
inline float hsum(__m256 x) {
    // split 256 bit vector into 128 bit, then use below hsum(__m128)
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    return hsum(_mm_add_ps(hi, lo));
}
#endif

// declaration of basic template function load
template <typename T> 
T load(const float *);

#if defined(__SSE__)
template <> 
inline __m128 load<__m128>(const float *p) {
    return _mm_loadu_ps(p);
}
#endif  // __SSE__

#if defined(__AVX2__)
template <> 
inline __m256 load<__m256>(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

// declaration of basic template function setzeros
template <typename T>
inline T setzeros();

#if defined(__SSE__)
template <>
inline __m128 setzeros<__m128>() { return _mm_setzero_ps(); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 setzeros<__m256>() { return _mm256_setzero_ps(); }
#endif

// declaration set1 basic template function
template <typename T>
inline T set1(float x);

#if defined(__SSE__)
template <>
inline __m128 set1<__m128>(float x) { return _mm_set1_ps(x); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 set1<__m256>(float x) { return _mm256_set1_ps(x); }
#endif



}
}
#endif // X86_KERNELS_HPP_