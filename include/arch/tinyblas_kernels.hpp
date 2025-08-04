#ifndef TINYBLAS_KERNELS_HPP_
#define TINYBLAS_KERNELS_HPP_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <new>

#if defined(__SSE__)
#include <smmintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h> 
#endif

namespace tinyBLAS {
namespace detail {

// 
#if defined(__SSE__)
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
inline __m128d add(__m128d x, __m128d y) { return _mm_add_pd(x, y); }
inline __m128d sub(__m128d x, __m128d y) { return _mm_sub_pd(x, y); }
inline __m128d mul(__m128d x, __m128d y) { return _mm_mul_pd(x, y); }
#endif  // __SSE4_2__

#if defined(__AVX2__)
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
inline __m256d add(__m256d x, __m256d y) { return _mm256_add_pd(x, y); }
inline __m256d sub(__m256d x, __m256d y) { return _mm256_sub_pd(x, y); }
inline __m256d mul(__m256d x, __m256d y) { return _mm256_mul_pd(x, y); }
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
inline __m256 madd<__m256>(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}

template<>
inline __m256d madd<__m256d>(__m256d a, __m256d b, __m256d c) {
    return _mm256_fmadd_pd(a, b, c);
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

inline double hsum(__m128d x) {
    __m128d x_shuffled = _mm_shuffle_pd(x, x, 0x01);
    __m128d sum = _mm_add_pd(x, x_shuffled);
    return _mm_cvtsd_f64(sum);
}
#endif

#if defined(__AVX2__)
inline float hsum(__m256 x) {
    // split 256 bit vector into 128 bit, then use below hsum(__m128)
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    return hsum(_mm_add_ps(hi, lo));
}

inline double hsum(__m256d x) {
    __m256d perm = _mm256_permute2f128_pd(x, x, 0x01);
    __m256d sum = _mm256_add_pd(perm, x);
    __m256d sums = _mm256_hadd_pd(sum, sum);
    return _mm256_cvtsd_f64(sums);
}
#endif

// declaration of basic template function load
template <typename T> 
T load(const float *);

template <typename T> 
T load(const double *);

#if defined(__SSE__)
template <> 
inline __m128 load<__m128>(const float *p) {
    return _mm_loadu_ps(p);
}
template <> 
inline __m128d load<__m128d>(const double *p) {
    return _mm_loadu_pd(p);
}
#endif  // __SSE__

#if defined(__AVX2__)
template <> 
inline __m256 load<__m256>(const float *p) {
    return _mm256_loadu_ps(p);
}

template <> 
inline __m256d load<__m256d>(const double *p) {
    return _mm256_loadu_pd(p);
}
#endif // __AVX__

// declaration of basic template function setzeros
template <typename T>
inline T setzeros();

#if defined(__SSE__)
template <>
inline __m128 setzeros<__m128>() { return _mm_setzero_ps(); }

template <>
inline __m128d setzeros<__m128d>() { return _mm_setzero_pd(); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 setzeros<__m256>() { return _mm256_setzero_ps(); }

template <>
inline __m256d setzeros<__m256d>() { return _mm256_setzero_pd(); }
#endif

// declaration set1 basic template function
template <typename T>
inline T set1(float x);

template <typename T>
inline T set1(double x);

#if defined(__SSE__)
template <>
inline __m128 set1<__m128>(float x) { return _mm_set1_ps(x); }
template <>
inline __m128d set1<__m128d>(double x) { return _mm_set1_pd(x); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 set1<__m256>(float x) { return _mm256_set1_ps(x); }
template <>
inline __m256d set1<__m256d>(double x) { return _mm256_set1_pd(x); }
#endif

#if defined(__SSE__)
inline void store(float *a, __m128 b) { _mm_storeu_ps(a, b); }
inline void store(double *a, __m128d b) { _mm_storeu_pd(a, b); }
#endif

#if defined(__AVX2__)
inline void store(float *a, __m256 b) { _mm256_storeu_ps(a, b); }
inline void store(double *a, __m256d b) { _mm256_storeu_pd(a, b); }
#endif

#if defined(__AVX2__)
inline __m256 broadcast(float *x) { return _mm256_broadcast_ss(x); } 
inline __m256d broadcast(double *x) { return _mm256_broadcast_sd(x); }
#endif

#if defined(__SSE__)
inline __m128  shuffle(__m128 a, __m128 b, int imm8) { 
    return _mm_shuffle_ps(a, b, imm8); 
} 
inline __m128d  shuffle(__m128d a, __m128d b, int imm8) { 
    return _mm_shuffle_pd(a, b, imm8); 
}
#endif

#if defined(__AVX2__)
inline __m256 shuffle(__m256 a, __m256 b, const int imm8) { 
    return _mm256_shuffle_ps(a, b, imm8); 
} 
inline __m256d shuffle(__m256d a, __m256d b, const int imm8) { 
    return _mm256_shuffle_pd(a, b, imm8); 
}
#endif

template <typename TA, typename TB, typename TC, int64_t RM, int64_t RN>
using MicroKernelType = std::function<void(int64_t, TA*, TB*, TC*, int64_t)>;

template <typename T>
struct MicroKernelCtx {
    T *next;
    char *flag;
    int64_t k;
    int64_t m;
    int64_t n;
};

template <typename T>
using MicroKernelCtxType = MicroKernelCtx<T>;

} // namespace detail
} // namespace tinyBLAS

#endif // TINYBLAS_GEMM_KERNELS_HPP_