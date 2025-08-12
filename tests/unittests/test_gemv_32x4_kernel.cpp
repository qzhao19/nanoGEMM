#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <tinyblas_gemv.hpp>

// A(i, j)     A[(j)*lda + (i)]
// B(i, j)     B[(j)*ldb + (i)]
// C(i, j)     C[(j)*ldc + (i)]
template <typename T>
class GEMV32x4KernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        engine_.seed(42);
        tol_ = 1e-4;
    }

    void generate_test_data(int64_t m, int64_t n, int64_t lda, T min = -1, T max = 1) {
        A = new T[n * lda];
        x = new T[n];
        y = new T[m];
        y_ref = new T[m];

        int64_t i, j, p;
        std::uniform_real_distribution<T> dist(min, max);

        for (p = 0; p < n; ++p) {
            for (i = 0; i < m; ++i) {
                A[p * lda + i] = dist(engine_);
            }
        }

        for (j = 0; j < n; ++j) {
            x[j] = dist(engine_);
        }

        for (int64_t i = 0; i < m; ++i) {
            y[i] = static_cast<T>(0.0);
            y_ref[i] = static_cast<T>(0.0);
        }
    }

    void matmul_ref(int64_t m, int64_t n, T *A, int64_t lda, T *x, T *y_ref) {
        int64_t i, j, p;
        for (int64_t i = 0; i < m; ++i) {
            T sum = 0;
            for (int64_t j = 0; j < n; ++j) {
                sum += A[j * lda + i] * x[j];
            }
            y_ref[i] = sum;
        }
    }

    bool compute_error(int64_t m, T *y, T *y_ref) {
        bool hasError = false;
        for (int64_t i = 0; i < m; ++i) {
            T diff = std::fabs(y[i] - y_ref[i]);
            if (diff > tol_) {
                std::printf("y[ %ld ] != y_ref, %E, %E\n", i, y[i], y_ref[i]);
                hasError = true;
            }
        }
        return hasError;
    }

    void TearDown() override {
        // release memory
        delete[] A;
        delete[] x;
        delete[] y;
        delete[] y_ref;
    }

    std::mt19937 engine_;
    T *A = nullptr;
    T *x = nullptr;
    T *y = nullptr;
    T *y_ref = nullptr;
    T tol_;
};

using TestTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(GEMV32x4KernelTest, TestTypes);

TYPED_TEST(GEMV32x4KernelTest, BasicMultiplyOddSize) {
    using T = TypeParam;
    int64_t M = 251, N = 173;
    int64_t lda = M;
    this->generate_test_data(M, N, lda);

    // naive GEMM for reference
    this->matmul_ref(M, N, this->A, lda, this->x, this->y_ref);

    // Use matmul API function instead of directly instantiating GEMM
    tinyBLAS::matmul(M, N, this->A, lda, this->x, this->y_ref);

    bool errorFound = this->compute_error(M, this->y, this->y_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";
}

TYPED_TEST(GEMV32x4KernelTest, BasicMultiplyEvenSize) {
    using T = TypeParam;
    int64_t M = 256, N = 128;
    int64_t lda = M;
    this->generate_test_data(M, N, lda);

    // naive GEMM for reference
    this->matmul_ref(M, N, this->A, lda, this->x, this->y_ref);

    // Use matmul API function instead of directly instantiating GEMM
    tinyBLAS::matmul(M, N, this->A, lda, this->x, this->y);

    bool errorFound = this->compute_error(M, this->y, this->y_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";
}