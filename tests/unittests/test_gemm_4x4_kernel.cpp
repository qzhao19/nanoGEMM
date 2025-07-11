#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <random>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <include/gemm.hpp>

template<typename T>
class GEMM8x4KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_.seed(42);
        tol_ = 1E-10;
    }

    T* generate_test_data(std::size_t size, T min = -1, T max = 1) {
        std::uniform_real_distribution<T> dist(min, max);
        T* arr = new T[size];
        for (std::size_t i = 0; i < size; ++i) {
            arr[i] = dist(engine_);
        }
        return arr;
    }

    bool compute_error(
        int ldc,
        int ldc_ref,
        int m,
        int n,
        T *C,
        T *C_ref) {

        bool hasError = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                T diff = std::fabs(C[i + j * ldc] - C_ref[i + j * ldc_ref]);
                if (diff > tol_) {
                    std::printf("C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C[i + j * ldc], C_ref[i + j * ldc_ref]);
                    hasError = true;
                }
            }
        }
        return hasError;
    }

    void TearDown() override {
        // release memory
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_ref;
    }

    std::mt19937 engine_;
    T* A = nullptr;
    T* B = nullptr;
    T* C = nullptr;
    T* C_ref = nullptr;
    T tol_;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GEMM8x4KernelTest, TestTypes);

TYPED_TEST(GEMM8x4KernelTest, BasicMultiply) {
    using T = TypeParam;
    constexpr int64_t M = 8, N = 4, K = 8;
    this->A = this->generate_test_data(M * K);
    this->B = this->generate_test_data(K * N);
    this->C = new T[M * N]();
    this->C_ref = new T[M * N]();

    // naive GEMM for reference
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k)
                this->C_ref[m + n * M] += A[m + k * M] * B[k + n * K];

    // GEMM class
    gemm::GEMM<T, T, T, 8, 4> gemm(
        this->A, M, this->B, K, this->C, M
    );
    gemm.multiply(M, N, K);

    bool errorFound = this->compute_error(M, M, M, N, this->C, this->C_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";

}
