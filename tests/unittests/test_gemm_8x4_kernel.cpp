#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <random>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <gemm.hpp>

// A(i, j)     A[(j)*lda + (i)]
// B(i, j)     B[(j)*ldb + (i)]
// C(i, j)     C[(j)*ldc + (i)]
template<typename T>
class GEMM8x4KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_.seed(42);
        tol_ = 1e-4;
    }

    void generate_test_data(
        int64_t m,
        int64_t n,
        int64_t k, 
        int64_t lda,
        int64_t ldb,
        int64_t ldc,
        T min = -1, T max = 1) {
        
        A = new T[k * lda];
        B = new T[n * ldb];
        C = new T[n * ldc];
        C_ref = new T[n * ldc];

        int64_t i, j, p;
        std::uniform_real_distribution<T> dist(min, max);

        for (p = 0; p < k; ++p) {
            for (i = 0; i < m; ++i) {
                A[p * lda + i] = dist(engine_);	
            }
        }

        for (j = 0; j < n; ++j) {
            for (p = 0; p < k; ++p) {
                B[j * ldb + p] = dist(engine_);
            }
        }

        for (j = 0; j < n; ++j) {
            for (i = 0; i < m; ++i) {
                C_ref[j * ldc + i] = static_cast<T>(0.0);	
                    C[j * ldc + i] = static_cast<T>(0.0);	
            }
        }
    }

    void matmul_ref(
        int64_t m,
        int64_t n,
        int64_t k,
        T *A,
        int64_t lda,
        T *B,
        int64_t ldb,
        T *C,
        int64_t ldc) {
        
        int64_t i, j, p;
        for (i = 0; i < m; i ++) {
            for (j = 0; j < n; j ++) {
                for (p = 0; p < k; p ++) {
                    C[j * ldc + i] += A[p * lda + i] * B[j * ldb + p];
                }
            }
        }
    }

    bool compute_error(
        int64_t ldc,
        int64_t ldc_ref,
        int64_t m,
        int64_t n,
        T *C,
        T *C_ref) {

        bool hasError = false;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                T diff = std::fabs(C[i + j * ldc] - C_ref[i + j * ldc_ref]);
                if (diff > tol_) {
                    std::printf("C[ %ld ][ %ld ] != C_ref, %E, %E\n", i, j, C[i + j * ldc], C_ref[i + j * ldc_ref]);
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

TYPED_TEST(GEMM8x4KernelTest, BasicMultiplyOddSize) {
    using T = TypeParam;
    int64_t M = 251, N = 173, K = 251;
    int64_t lda = M, ldb = K, ldc = M;
    this->generate_test_data(M, N, K, lda, ldb, ldc);
    
    // naive GEMM for reference
    this->matmul_ref(M, N, K, this->A, lda, this->B, ldb, this->C_ref, ldc);

    // Use matmul API function instead of directly instantiating GEMM
    gemm::matmul(M, N, K, this->A, lda, this->B, ldb, this->C, ldc);
    
    bool errorFound = this->compute_error(M, M, M, N, this->C, this->C_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";
}


TYPED_TEST(GEMM8x4KernelTest, BasicMultiplyEvenSize) {
    using T = TypeParam;
    int64_t M = 256, N = 128, K = 256;
    int64_t lda = M, ldb = K, ldc = M;
    this->generate_test_data(M, N, K, lda, ldb, ldc);
    
    // naive GEMM for reference
    this->matmul_ref(M, N, K, this->A, lda, this->B, ldb, this->C_ref, ldc);

    // Use matmul API function instead of directly instantiating GEMM
    gemm::matmul(M, N, K, this->A, lda, this->B, ldb, this->C, ldc);
    
    bool errorFound = this->compute_error(M, M, M, N, this->C, this->C_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";
}