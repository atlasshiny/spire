#pragma once
#include <spire/activation_payload.hpp>
#include <spire/cuda_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace spire {
constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status_ = (call);                                       \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\\n", __FILE__,      \
                         __LINE__, static_cast<int>(status_));                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

inline cublasHandle_t thread_handle() {
    struct Holder {
        cublasHandle_t handle = nullptr;
        Holder() { if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) std::exit(EXIT_FAILURE); }
        ~Holder() { if (handle != nullptr) cublasDestroy(handle); }
    };
    thread_local Holder holder;
    return holder.handle;
}

inline unsigned int blocks_for(std::size_t count) {
    return static_cast<unsigned int>(std::min<std::size_t>((count + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
}

inline bool fits_int(std::size_t value) { return value <= static_cast<std::size_t>(std::numeric_limits<int>::max()); }

inline void* thread_scratch(std::size_t bytes) {
    struct Cache { void* ptr = nullptr; std::size_t bytes = 0; ~Cache() { if (ptr) cudaFree(ptr); } };
    thread_local Cache cache;
    if (bytes > cache.bytes) {
        if (cache.ptr) cudaFree(cache.ptr);
        CUDA_CHECK(cudaMalloc(&cache.ptr, bytes));
        cache.bytes = bytes;
    }
    return cache.ptr;
}

__global__ void bias_relu_f32_kernel(float* __restrict__ out, const float* __restrict__ bias, std::size_t total, std::size_t cols);
__global__ void bias_add_bf16_kernel(__nv_bfloat16* __restrict__ out, const float* __restrict__ bias, std::size_t total, std::size_t cols);
__global__ void f32_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, std::size_t count);

enum class GemmBackend { Auto, Kernel, Gemm };
inline GemmBackend backend_override() {
    struct Once {
        GemmBackend value = GemmBackend::Auto;
        Once() {
            const char* env = std::getenv("SPIRE_SAE_BACKEND");
            if (!env) return;
            std::string_view mode(env);
            if (mode == "kernel") value = GemmBackend::Kernel;
            else if (mode == "gemm") value = GemmBackend::Gemm;
        }
    };
    static const Once once;
    return once.value;
}

constexpr std::size_t kEncoderGemmMinBatch = 2;
constexpr std::size_t kDecoderGemmMinBatch = 1;
inline bool use_gemm(std::size_t batch_size, std::size_t min_batch) {
    switch (backend_override()) {
        case GemmBackend::Kernel: return false;
        case GemmBackend::Gemm: return true;
        default: return batch_size >= min_batch;
    }
}
} // namespace spire
