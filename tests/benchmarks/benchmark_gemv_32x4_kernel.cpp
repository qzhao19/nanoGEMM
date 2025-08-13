#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <tinyblas_gemv.hpp>

// Reference GEMV implementation for benchmarking
template <typename T>
void gemv_ref(int64_t m, int64_t n, const T* A, int64_t lda, const T* x, T* y) {
    for (int64_t i = 0; i < m; ++i) {
        T sum = static_cast<T>(0);
        for (int64_t j = 0; j < n; ++j) {
            sum += A[j * lda + i] * x[j];
        }
        y[i] = sum;
    }
}

template <typename T>
class GEMV32x4Benchmark {
public:
    GEMV32x4Benchmark(int64_t m, int64_t n)
        : m_(m), n_(n), lda_(m) {
        engine_.seed(42);
        allocate_and_generate_data();
    }

    ~GEMV32x4Benchmark() {
        delete[] A_;
        delete[] x_;
        delete[] y_;
    }

    void run_benchmark() {
        // Run the actual GEMV operation
        tinyBLAS::matmul(m_, n_, A_, lda_, x_, y_);
    }

    void run_benchmark_ref() {
        gemv_ref(m_, n_, A_, lda_, x_, y_);
    }

private:
    void allocate_and_generate_data() {
        A_ = new T[n_ * lda_];
        x_ = new T[n_];
        y_ = new T[m_];

        std::uniform_real_distribution<T> dist(-1.0, 1.0);

        // Initialize matrix A (column-major)
        for (int64_t j = 0; j < n_; ++j) {
            for (int64_t i = 0; i < m_; ++i) {
                A_[j * lda_ + i] = dist(engine_);
            }
        }
        // Initialize vector x
        for (int64_t j = 0; j < n_; ++j) {
            x_[j] = dist(engine_);
        }
        // Initialize vector y
        for (int64_t i = 0; i < m_; ++i) {
            y_[i] = static_cast<T>(0);
        }
    }

    int64_t m_;
    int64_t n_;
    int64_t lda_;
    T* A_ = nullptr;
    T* x_ = nullptr;
    T* y_ = nullptr;
    std::mt19937 engine_;
};

// Benchmark function for optimized GEMV implementation
template <typename T>
static void BM_GEMV(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t n = state.range(1);

    GEMV32x4Benchmark<T> benchmark(m, n);

    for (auto _ : state) {
        benchmark.run_benchmark();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2 * m * n);
    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * m * n / 1e9, benchmark::Counter::kIsRate);
}

// Benchmark function for reference GEMV implementation
template <typename T>
static void BM_GEMV_Ref(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t n = state.range(1);

    GEMV32x4Benchmark<T> benchmark(m, n);

    for (auto _ : state) {
        benchmark.run_benchmark_ref();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2 * m * n);
    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * m * n / 1e9, benchmark::Counter::kIsRate);
}

// Define benchmark test cases for float (optimized)
BENCHMARK_TEMPLATE(BM_GEMV, float)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({251, 173})
    ->Args({1000, 100})
    ->Unit(benchmark::kMillisecond);

// Define benchmark test cases for float (reference)
BENCHMARK_TEMPLATE(BM_GEMV_Ref, float)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({251, 173})
    ->Unit(benchmark::kMillisecond);

// Define benchmark test cases for double (optimized)
// BENCHMARK_TEMPLATE(BM_GEMV, double)
//     ->Args({256, 256})
//     ->Args({512, 512})
//     ->Args({1024, 1024})
//     ->Args({251, 173})
//     ->Args({1000, 100})
//     ->Unit(benchmark::kMillisecond);

// Define benchmark test cases for double (reference)
// BENCHMARK_TEMPLATE(BM_GEMV_Ref, double)
//     ->Args({256, 256})
//     ->Args({512, 512})
//     ->Args({251, 173})
//     ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();