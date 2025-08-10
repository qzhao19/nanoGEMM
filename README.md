# tinyBLAS

tinyBLAS provides high-performance GEMM (matrix-matrix multiplication) and GEMV (matrix-vector multiplication) algorithms, designed specifically to accelerate matrix operations in large language model (LLM) inference workloads. The library features adaptive micro-kernels and CPU-aware optimizations for modern processors.

## Features

- **Optimized Micro-Kernels**: Hand-tuned for SIMD instruction sets (SSE, AVX2, FMA), with automatic kernel selection based on CPU features.
- **Blocking and Packing**: Efficient data packing and blocking strategies to maximize cache and memory bandwidth utilization.
- **Multi-threading**: OpenMP-based parallelism for full multi-core CPU utilization.
- **Extensible Design**: Easily add custom micro-kernels and support for multiple data types (float, double, etc.).
- **Reference Implementations & Benchmarks**: Includes simple reference GEMM/GEMV and Google Benchmark-based performance tests for validation and comparison.

