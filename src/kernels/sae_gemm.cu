#include <spire/kernels.cuh>
#include <spire/cuda_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status_ = (call);                                       \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,      \
                         __LINE__, static_cast<int>(status_));                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// One cuBLAS handle per calling thread, created on first use. This keeps the
// daemon thread and any synchronous caller independent (cuBLAS handles are not
// safe for concurrent use) without a global lock on the hot path.
cublasHandle_t thread_handle() {
    struct Holder {
        cublasHandle_t handle = nullptr;
        Holder() {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "cuBLAS handle creation failed\n");
                std::exit(EXIT_FAILURE);
            }
        }
        ~Holder() {
            if (handle != nullptr) {
                cublasDestroy(handle);
            }
        }
    };
    thread_local Holder holder;
    return holder.handle;
}

unsigned int blocks_for(std::size_t count) {
    const std::size_t needed = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return static_cast<unsigned int>(
        std::min<std::size_t>(needed, kMaxBlocks));
}

bool fits_int(std::size_t value) {
    return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

// Per-thread scratch for the decoder's FP32->BF16 conversion. Allocated once and
// only grown on demand, so the hot path never calls cudaMalloc. A thread's
// launches are stream-ordered, so reusing the buffer across calls is safe.
void* thread_scratch(std::size_t bytes) {
    struct Cache {
        void* ptr = nullptr;
        std::size_t bytes = 0;
        ~Cache() {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
    };
    thread_local Cache cache;
    if (bytes > cache.bytes) {
        if (cache.ptr != nullptr) {
            CUDA_CHECK(cudaFree(cache.ptr));
            cache.ptr = nullptr;
            cache.bytes = 0;
        }
        CUDA_CHECK(cudaMalloc(&cache.ptr, bytes));
        cache.bytes = bytes;
    }
    return cache.ptr;
}

// Encoder epilogue: out = ReLU(out + bias[col]).
__global__ void bias_relu_f32_kernel(float* __restrict__ out,
                                     const float* __restrict__ bias,
                                     std::size_t total,
                                     std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        out[i] = fmaxf(0.0f, out[i] + bias[i % cols]);
    }
}

// Decoder epilogue: out = out + bias[col], in BF16.
__global__ void bias_add_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                     const float* __restrict__ bias,
                                     std::size_t total,
                                     std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        out[i] = __float2bfloat16(__bfloat162float(out[i]) + bias[i % cols]);
    }
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ src,
                                   __nv_bfloat16* __restrict__ dst,
                                   std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

} // namespace

namespace spire {

void launch_sae_kernel_gemm(const ActivationPayload& payload) {
    if (!payload.d_input || !payload.d_weights || !payload.d_bias ||
        !payload.d_latents || payload.batch_size == 0 ||
        payload.hidden_dim == 0 || payload.sae_dim == 0) {
        return;
    }
    if (!fits_int(payload.batch_size) || !fits_int(payload.hidden_dim) ||
        !fits_int(payload.sae_dim)) {
        return;
    }

    const int batch = static_cast<int>(payload.batch_size);
    const int hidden = static_cast<int>(payload.hidden_dim);
    const int sae = static_cast<int>(payload.sae_dim);

    cublasHandle_t handle = thread_handle();
    CUBLAS_CHECK(cublasSetStream(handle, payload.stream));

    // Row-major C[B,S] = x[B,H] * W[S,H]^T, computed as the equivalent
    // column-major GEMM C_col[S,B] = W_row[S,H] * x_row[B,H]^T. The output
    // buffer is C_col [S,B] with ld=S, which is byte-identical to row-major
    // C [B,S].
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, sae, batch, hidden,
        &alpha,
        payload.d_weights.get(), CUDA_R_16BF, hidden,
        payload.d_input.get(), CUDA_R_16BF, hidden,
        &beta,
        payload.d_latents.get(), CUDA_R_32F, sae,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // Fused epilogue equivalent: z = ReLU(z + b_enc).
    const std::size_t total = payload.batch_size * payload.sae_dim;
    bias_relu_f32_kernel<<<blocks_for(total), kThreadsPerBlock, 0,
                           payload.stream>>>(
        payload.d_latents.get(), payload.d_bias.get(), total,
        payload.sae_dim);
    CUDA_CHECK(cudaGetLastError());
}

void launch_sae_decoder_gemm(const float* latents,
                             const __nv_bfloat16* weights,
                             const float* bias,
                             __nv_bfloat16* output,
                             std::size_t batch_size,
                             std::size_t sae_dim,
                             std::size_t hidden_dim,
                             cudaStream_t stream) {
    if (!latents || !weights || !bias || !output || batch_size == 0 ||
        sae_dim == 0 || hidden_dim == 0) {
        return;
    }
    if (!fits_int(batch_size) || !fits_int(sae_dim) || !fits_int(hidden_dim)) {
        return;
    }

    const int batch = static_cast<int>(batch_size);
    const int sae = static_cast<int>(sae_dim);
    const int hidden = static_cast<int>(hidden_dim);

    // cuBLAS requires both GEMM inputs to share a type; convert the FP32 latents
    // to BF16 in a reusable per-thread scratch buffer. The conversion kernel and
    // the GEMM are stream-ordered, so the same buffer can be reused on the next
    // launch from this thread without waiting on the host.
    auto* z = static_cast<__nv_bfloat16*>(
        thread_scratch(batch_size * sae_dim * sizeof(__nv_bfloat16)));
    f32_to_bf16_kernel<<<blocks_for(batch_size * sae_dim), kThreadsPerBlock, 0,
                         stream>>>(latents, z, batch_size * sae_dim);
    CUDA_CHECK(cudaGetLastError());

    cublasHandle_t handle = thread_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Row-major C[B,H] = z[B,S] * W[S,H], computed as column-major
    // C_col[H,B] = W_row[S,H]^T * z_row[B,S]^T. Output is C_col [H,B] ld=H,
    // byte-identical to row-major C [B,H].
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden, batch, sae,
        &alpha,
        weights, CUDA_R_16BF, hidden,
        z, CUDA_R_16BF, sae,
        &beta,
        output, CUDA_R_16BF, hidden,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaGetLastError());

    const std::size_t total = batch_size * hidden_dim;
    bias_add_bf16_kernel<<<blocks_for(total), kThreadsPerBlock, 0, stream>>>(
        output, bias, total, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace spire
