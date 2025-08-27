#ifndef KERNEL_FACTORY_HPP_
#define KERNEL_FACTORY_HPP_

#include <arch/tinyblas_kernels.hpp>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyBLAS {
namespace detail {

// forward definition
template <typename TA,
          typename TB,
          typename TC,
          int64_t RM,
          int64_t RN,
          int64_t CM,
          int64_t CK,
          int64_t CN>
class GEMM;

// abstract GEMM executor interface
template <typename TA, typename TB, typename TC>
class GEMMExecutor {
   public:
    virtual ~GEMMExecutor() = default;
    virtual void multiply(int64_t m,
                          int64_t n,
                          int64_t k,
                          const TA *A,
                          int64_t lda,
                          const TB *B,
                          int64_t ldb,
                          TC *C,
                          int64_t ldc) = 0;
};

// concrete implementation of the GEMM executor
template <typename TA,
          typename TB,
          typename TC,
          int64_t RM,
          int64_t RN,
          int64_t CM,
          int64_t CK,
          int64_t CN>
class GEMMExecutorImpl : public GEMMExecutor<TA, TB, TC> {
   private:
    GEMMMicroKernelType<TA, TB, TC, RM, RN> kernel_;

   public:
    GEMMExecutorImpl(GEMMMicroKernelType<TA, TB, TC, RM, RN> kernel) : kernel_(kernel) {}

    void multiply(int64_t m,
                  int64_t n,
                  int64_t k,
                  const TA *A,
                  int64_t lda,
                  const TB *B,
                  int64_t ldb,
                  TC *C,
                  int64_t ldc) override {
        GEMM<TA, TB, TC, RM, RN, CM, CK, CN> gemm_engine(A, lda, B, ldb, C, ldc, kernel_);
        gemm_engine.multiply(m, n, k);
    }
};

// factory class
template <typename TA, typename TB, typename TC>
class KernelFactory {
   public:
    using ExecutorPtr = std::unique_ptr<GEMMExecutor<TA, TB, TC>>;
    using CreatorFunc = std::function<ExecutorPtr()>;

   private:
    std::unordered_map<std::string, CreatorFunc> creators_;
    std::string default_kernel_ = "4x4";

   public:
    static KernelFactory &get_instance() {
        static KernelFactory instance;
        return instance;
    }

    // registery kernel
    template <int64_t RM, int64_t RN, int64_t CM, int64_t CK, int64_t CN>
    void register_kernel(const std::string &name, GEMMMicroKernelType<TA, TB, TC, RM, RN> kernel) {
        creators_[name] = [kernel]() -> ExecutorPtr {
            return std::make_unique<GEMMExecutorImpl<TA, TB, TC, RM, RN, CM, CK, CN>>(kernel);
        };
    }

    // create kernel executor
    ExecutorPtr create_executor(const std::string &kernel_name) {
        auto it = creators_.find(kernel_name);
        if (it != creators_.end()) {
            return it->second();
        }

        if (kernel_name == "auto") {
            return create_auto_executor();
        }

        // use default kernel
        auto default_it = creators_.find(default_kernel_);
        if (default_it != creators_.end()) {
            return default_it->second();
        }

        throw std::runtime_error("No suitable kernel found for: " + kernel_name);
    }

    // check
    bool has_kernel(const std::string &name) const {
        return creators_.find(name) != creators_.end();
    }

    // get all available kernel names
    std::vector<std::string> get_available_kernels() const {
        std::vector<std::string> kernels;
        for (const auto &pair : creators_) {
            kernels.push_back(pair.first);
        }
        return kernels;
    }

   private:
    KernelFactory() {
        // register the default kernel in the constructor
        register_default_kernels();
    }

    void register_default_kernels();

    ExecutorPtr create_auto_executor() {
        if (!has_kernel(default_kernel_)) {
            throw std::runtime_error("No kernels available.");
        }

        // default use 4x4 kernel
        return creators_[default_kernel_]();
    }
};

}  // namespace detail
}  // namespace tinyBLAS

#endif  // KERNEL_FACTORY_HPP_