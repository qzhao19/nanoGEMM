#include <benchmark/benchmark.h>
#include <iostream>
#include <random>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <gemm.hpp>

// Reference matrix multiplication implementation for benchmarking
template<typename T>
void matmul_ref(int64_t m, int64_t n, int64_t k, 
                const T* A, int64_t lda, 
                const T* B, int64_t ldb,
                T* C, int64_t ldc) {
    // Simple triple-loop implementation
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            T sum = static_cast<T>(0.0);
            for (int64_t p = 0; p < k; ++p) {
                sum += A[p * lda + i] * B[j * ldb + p];
            }
            C[j * ldc + i] = sum;
        }
    }
}

template<typename T>
class GEMM4x4Benchmark {
public:
    GEMM4x4Benchmark(int64_t m, int64_t n, int64_t k) 
        : m_(m), n_(n), k_(k), lda_(m), ldb_(k), ldc_(m) {
        // Initialize random number generator
        engine_.seed(42);
        allocate_and_generate_data();
    }

    ~GEMM4x4Benchmark() {
        delete[] A_;
        delete[] B_;
        delete[] C_;
    }

    void run_benchmark() {
        // Run the actual GEMM operation
        gemm::matmul(m_, n_, k_, A_, lda_, B_, ldb_, C_, ldc_);
    }

    // Add reference implementation method
    void run_benchmark_ref() {
        // Run the reference GEMM operation
        matmul_ref(m_, n_, k_, A_, lda_, B_, ldb_, C_, ldc_);
    }

private:
    void allocate_and_generate_data() {
        A_ = new T[k_ * lda_];
        B_ = new T[n_ * ldb_];
        C_ = new T[n_ * ldc_];

        std::uniform_real_distribution<T> dist(-1.0, 1.0);

        // Initialize matrices with random values
        for (int64_t p = 0; p < k_; ++p) {
            for (int64_t i = 0; i < m_; ++i) {
                A_[p * lda_ + i] = dist(engine_);
            }
        }

        for (int64_t j = 0; j < n_; ++j) {
            for (int64_t p = 0; p < k_; ++p) {
                B_[j * ldb_ + p] = dist(engine_);
            }
        }

        for (int64_t j = 0; j < n_; ++j) {
            for (int64_t i = 0; i < m_; ++i) {
                C_[j * ldc_ + i] = static_cast<T>(0.0);
            }
        }
    }

    int64_t m_;
    int64_t n_;
    int64_t k_;
    int64_t lda_;
    int64_t ldb_;
    int64_t ldc_;
    T* A_ = nullptr;
    T* B_ = nullptr;
    T* C_ = nullptr;
    std::mt19937 engine_;
};

// Benchmark function for optimized GEMM implementation
template<typename T>
static void BM_GEMM(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);
    
    GEMM4x4Benchmark<T> benchmark(m, n, k);
    
    // Perform setup
    for (auto _ : state) {
        // This code gets timed
        benchmark.run_benchmark();
    }
    
    // Report stats
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2 * m * n * k);
    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * m * n * k / 1e9,
        benchmark::Counter::kIsRate);
}

// Benchmark function for reference GEMM implementation
template<typename T>
static void BM_GEMM_Ref(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);
    
    GEMM4x4Benchmark<T> benchmark(m, n, k);
    
    // Perform setup
    for (auto _ : state) {
        // This code gets timed
        benchmark.run_benchmark_ref();
    }
    
    // Report stats
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2 * m * n * k);
    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * m * n * k / 1e9,
        benchmark::Counter::kIsRate);
}

// Define benchmark test cases for float (optimized)
BENCHMARK_TEMPLATE(BM_GEMM, float)
    ->Args({256, 256, 256})  // Square matrix
    ->Args({512, 512, 512})  // Larger square matrix
    ->Args({1024, 1024, 1024})  // Even larger
    ->Args({251, 173, 251})  // Odd-sized matrix
    ->Args({1000, 100, 500})  // Rectangular matrix
    ->Unit(benchmark::kMillisecond);

// Define benchmark test cases for float (reference)
BENCHMARK_TEMPLATE(BM_GEMM_Ref, float)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({251, 173, 251})  // Odd-sized matrix to compare with optimized
    ->Unit(benchmark::kMillisecond);

// Define benchmark test cases for double (optimized)
BENCHMARK_TEMPLATE(BM_GEMM, double)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({251, 173, 251})
    ->Args({1000, 100, 500})
    ->Unit(benchmark::kMillisecond);

// Define benchmark test cases for double (reference)
BENCHMARK_TEMPLATE(BM_GEMM_Ref, double)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({251, 173, 251})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();