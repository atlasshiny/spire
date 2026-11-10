#include <spire/kernels.cuh>
#include <spire/cuda_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;

__device__ __forceinline__ float2 bf162_scale(float value, __nv_bfloat162 weights) {
    const float2 wf = __bfloat1622float2(weights);
    return make_float2(value * wf.x, value * wf.y);
}

__device__ __forceinline__ float2 pair_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// Vectorized SAE decode: x' = z * W_dec + b_dec.
// One thread produces two adjacent hidden outputs. The decoder weight row is
// loaded as a packed bf162, so consecutive lanes read consecutive pairs
// (coalesced 128 B warp transactions) and the k-loop is unrolled 4x for
// instruction-level parallelism — mirroring the encoder's inner loop.
__global__ __launch_bounds__(kThreadsPerBlock) void sae_decode_pairs_kernel(
    const float* __restrict__ latents,          // [batch_size, sae_dim]
    const __nv_bfloat16* __restrict__ weights,  // [sae_dim, hidden_dim], row-major
    const float* __restrict__ bias,             // [hidden_dim]
    __nv_bfloat16* __restrict__ output,         // [batch_size, hidden_dim]
    size_t hidden_dim,
    size_t sae_dim,
    size_t total_pairs
) {
    const size_t pairs_per_row = hidden_dim >> 1;
    const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t thread_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t pair_idx = thread; pair_idx < total_pairs; pair_idx += thread_stride) {
        const size_t batch_idx = pair_idx / pairs_per_row;
        const size_t h_idx = (pair_idx % pairs_per_row) * 2;

        const float* z = latents + batch_idx * sae_dim;
        const __nv_bfloat16* w_pair = weights + h_idx;

        float2 sum = make_float2(0.0f, 0.0f);

        size_t k = 0;
        for (; k + 3 < sae_dim; k += 4) {
            sum = pair_add(sum, bf162_scale(
                z[k + 0], *reinterpret_cast<const __nv_bfloat162*>(
                              w_pair + (k + 0) * hidden_dim)));
            sum = pair_add(sum, bf162_scale(
                z[k + 1], *reinterpret_cast<const __nv_bfloat162*>(
                              w_pair + (k + 1) * hidden_dim)));
            sum = pair_add(sum, bf162_scale(
                z[k + 2], *reinterpret_cast<const __nv_bfloat162*>(
                              w_pair + (k + 2) * hidden_dim)));
            sum = pair_add(sum, bf162_scale(
                z[k + 3], *reinterpret_cast<const __nv_bfloat162*>(
                              w_pair + (k + 3) * hidden_dim)));
        }
        for (; k < sae_dim; ++k) {
            sum = pair_add(sum, bf162_scale(
                z[k], *reinterpret_cast<const __nv_bfloat162*>(
                          w_pair + k * hidden_dim)));
        }

        const float2 b = make_float2(bias[h_idx], bias[h_idx + 1]);
        const size_t out = batch_idx * hidden_dim + h_idx;
        output[out] = __float2bfloat16(sum.x + b.x);
        output[out + 1] = __float2bfloat16(sum.y + b.y);
    }
}

// Fallback for odd hidden sizes, where two adjacent outputs cannot be packed
// into a bf162. One thread computes one output element.
__global__ __launch_bounds__(kThreadsPerBlock) void sae_decode_scalar_kernel(
    const float* __restrict__ latents,          // [batch_size, sae_dim]
    const __nv_bfloat16* __restrict__ weights,  // [sae_dim, hidden_dim], row-major
    const float* __restrict__ bias,             // [hidden_dim]
    __nv_bfloat16* __restrict__ output,         // [batch_size, hidden_dim]
    size_t hidden_dim,
    size_t sae_dim,
    size_t total_outputs
) {
    const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t thread_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t out_idx = thread; out_idx < total_outputs; out_idx += thread_stride) {
        const size_t batch_idx = out_idx / hidden_dim;
        const size_t h_idx = out_idx % hidden_dim;

        const float* z = latents + batch_idx * sae_dim;
        const __nv_bfloat16* w_col = weights + h_idx;

        float sum = 0.0f;
        for (size_t k = 0; k < sae_dim; ++k) {
            sum += z[k] * __bfloat162float(w_col[k * hidden_dim]);
        }

        output[out_idx] = __float2bfloat16(sum + bias[h_idx]);
    }
}

} // namespace

namespace spire::legacy {

void launch_sae_decoder(const float* latents,
                        const __nv_bfloat16* weights,
                        const float* bias,
                        __nv_bfloat16* output,
                        std::size_t batch_size,
                        std::size_t sae_dim,
                        std::size_t hidden_dim,
                        cudaStream_t stream) {
    if (!latents || !weights || !bias || !output ||
        batch_size == 0 || sae_dim == 0 || hidden_dim == 0) {
        return;
    }

    // Even hidden sizes use the bf162-vectorized kernel; odd sizes fall back to
    // one thread per output element.
    if (hidden_dim % 2 == 0) {
        const size_t total_pairs = batch_size * (hidden_dim >> 1);
        const size_t required_blocks =
            (total_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min(required_blocks, static_cast<std::size_t>(kMaxBlocks)));

        // Launch asynchronously on the model's CUDA stream.
        sae_decode_pairs_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            latents, weights, bias, output,
            hidden_dim, sae_dim, total_pairs);
    } else {
        const size_t total_outputs = batch_size * hidden_dim;
        const size_t required_blocks =
            (total_outputs + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min(required_blocks, static_cast<std::size_t>(kMaxBlocks)));

        sae_decode_scalar_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            latents, weights, bias, output,
            hidden_dim, sae_dim, total_outputs);
    }

    // Catches launch-time errors only; async execution errors surface on the
    // stream (or at the next synchronization point).
    CUDA_CHECK(cudaGetLastError());
}

} // namespace spire
