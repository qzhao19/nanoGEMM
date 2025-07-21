#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <random>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <gemm.hpp>

template<typename T>
class GEMM4x4KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_.seed(42);
        tol_ = 1e-4;
    }

    T* generate_test_data(
        std::size_t size, 
        T min = -1, T max = 1) {
        
        std::uniform_real_distribution<T> dist(min, max);
        T* arr = new T[size];
        for (std::size_t i = 0; i < size; ++i) {
            arr[i] = dist(engine_);
        }
        return arr;
    }

    void matmul_ref(
        int m,
        int n,
        int k,
        T *A,
        int lda,
        T *B,
        int ldb,
        T *C,
        int ldc) {
        
        int    i, j, p;
        for (i = 0; i < m; i ++) {
            for (j = 0; j < n; j ++) {
                for (p = 0; p < k; p ++) {
                    C[j * ldc + i] += A[p * lda + i] * B[j * ldb + p];
                }
            }
        }
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
TYPED_TEST_SUITE(GEMM4x4KernelTest, TestTypes);

TYPED_TEST(GEMM4x4KernelTest, BasicMultiply) {
    using T = TypeParam;
    int64_t M = 64, N = 64, K = 64;
    int64_t lda = M, ldb = K, ldc = M;
    this->A = this->generate_test_data(M * K);
    this->B = this->generate_test_data(K * N);
    this->C = new T[M * N]();
    this->C_ref = new T[M * N]();

    // naive GEMM for reference
    this->matmul_ref(M, N, K, this->A, lda, this->B, ldb, this->C_ref, ldc);

    // Use matmul API function instead of directly instantiating GEMM
    gemm::matmul(M, N, K, this->A, lda, this->B, ldb, this->C, ldc);
    
    bool errorFound = this->compute_error(M, M, M, N, this->C, this->C_ref);
    ASSERT_FALSE(errorFound) << "Errors found in the result.";
}
