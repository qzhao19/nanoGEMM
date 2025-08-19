#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <tinyblas.hpp>

// A(i, j)     A[(j)*lda + (i)]
// B(i, j)     B[(j)*ldb + (i)]
// C(i, j)     C[(j)*ldc + (i)]
template <typename T>
class GEMM4x4KernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        engine_.seed(42);
        tol_ = 1e-4;
    }

    void generate_test_data(int64_t m,
                            int64_t n,
                            int64_t k,
                            int64_t lda,
                            int64_t ldb,
                            int64_t ldc,
                            T min = -1,
                            T max = 1) {
        A = new T[m * lda];
        B = new T[k * ldb];
        C = new T[m * ldc];
        C_ref = new T[m * ldc];

        int64_t i, j, p;
        std::uniform_real_distribution<T> dist(min, max);

        for (i = 0; i < m; ++i) {
            for (p = 0; p < k; ++p) {
                A[i * lda + p] = dist(engine_);
            }
        }

        for (p = 0; p < k; ++p) {
            for (j = 0; j < n; ++j) {
                B[p * ldb + j] = dist(engine_);
            }
        }

        for (i = 0; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                C_ref[i * ldc + j] = static_cast<T>(0.0);
                C[i * ldc + j] = static_cast<T>(0.0);
            }
        }
    }

    void matmul_ref(
        int64_t m, int64_t n, int64_t k, T *A, int64_t lda, T *B, int64_t ldb, T *C, int64_t ldc) {
        int64_t i, j, p;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                for (p = 0; p < k; p++) {
                    // C[i,j] += A[i,p] * B[p,j]
                    C[i * ldc + j] += A[i * lda + p] * B[p * ldb + j];
                }
            }
        }
    }

    bool compute_error(int64_t ldc, int64_t ldc_ref, int64_t m, int64_t n, T *C, T *C_ref) {
        bool hasError = false;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                // C[i,j] = C[i*ldc + j]
                T diff = std::fabs(C[i * ldc + j] - C_ref[i * ldc_ref + j]);
                if (diff > tol_) {
                    std::printf("C[ %ld ][ %ld ] != C_ref, %E, %E\n",
                                i,
                                j,
                                C[i * ldc + j],
                                C_ref[i * ldc_ref + j]);
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
    T *A = nullptr;
    T *B = nullptr;
    T *C = nullptr;
    T *C_ref = nullptr;
    T tol_;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GEMM4x4KernelTest, TestTypes);

TYPED_TEST(GEMM4x4KernelTest, BasicMultiplyOddSize) {
    using T = TypeParam;
    int64_t M = 251, N = 173, K = 251;
    int64_t lda = K, ldb = N, ldc = N;
    this->generate_test_data(M, N, K, lda, ldb, ldc);

    // naive GEMM for reference
    this->matmul_ref(M, N, K, this->A, lda, this->B, ldb, this->C_ref, ldc);

    // Use matmul API function instead of directly instantiating GEMM
    tinyBLAS::matmul(M, N, K, this->A, lda, this->B, ldb, this->C, ldc, "4x4");

    bool errorFound = this->compute_error(ldc, ldc, M, N, this->C, this->C_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";
}

TYPED_TEST(GEMM4x4KernelTest, BasicMultiplyEvenSize) {
    using T = TypeParam;
    int64_t M = 256, N = 64, K = 256;
    int64_t lda = K, ldb = N, ldc = N;
    this->generate_test_data(M, N, K, lda, ldb, ldc);

    // naive GEMM for reference
    this->matmul_ref(M, N, K, this->A, lda, this->B, ldb, this->C_ref, ldc);

    // Use matmul API function instead of directly instantiating GEMM
    tinyBLAS::matmul(M, N, K, this->A, lda, this->B, ldb, this->C, ldc, "4x4");

    bool errorFound = this->compute_error(ldc, ldc, M, N, this->C, this->C_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";
}