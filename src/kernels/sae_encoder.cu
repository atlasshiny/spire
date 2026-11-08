#include <spire/kernels.cuh>
#include <spire/cuda_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr unsigned int kMaxBlocks = 65535;

// Warp-per-output SAE encode: z = ReLU(W_enc * x + b_enc).
// One warp computes one (batch, latent) dot product. Lanes stride along the
// row so consecutive lanes read consecutive floats (coalesced 128B loads).
__global__ void sae_encode_kernel(
    const __nv_bfloat16* __restrict__ input, // [batch_size, hidden_dim]
    const float* __restrict__ weights,       // [sae_dim, hidden_dim], row-major
    const float* __restrict__ bias,          // [sae_dim]
    float* __restrict__ output,              // [batch_size, sae_dim]
    size_t hidden_dim,
    size_t sae_dim,
    size_t total_outputs
) {
    const unsigned int lane = threadIdx.x & 31u;
    const size_t global_warp =
        static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    const size_t warp_stride = static_cast<size_t>(gridDim.x) * kWarpsPerBlock;

    for (size_t out_idx = global_warp; out_idx < total_outputs; out_idx += warp_stride) {
        const size_t batch_idx = out_idx / sae_dim;
        const size_t latent_idx = out_idx % sae_dim;

        const __nv_bfloat16* x = input + batch_idx * hidden_dim;
        const float* w_row = weights + latent_idx * hidden_dim;

        float sum = 0.0f;
        for (size_t i = lane; i < hidden_dim; i += 32) {
            sum += w_row[i] * __bfloat162float(x[i]);
        }

        // Reduce the 32 partial sums; all lanes are active here.
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
        }

        if (lane == 0) {
            output[out_idx] = fmaxf(0.0f, sum + bias[latent_idx]);
        }
    }
}

} // namespace

namespace spire {

void launch_sae_kernel(const ActivationPayload& payload) {
    if (payload.d_input == nullptr || payload.d_weights == nullptr ||
        payload.d_bias == nullptr || payload.d_latents == nullptr ||
        payload.batch_size == 0 || payload.hidden_dim == 0 || payload.sae_dim == 0) {
        return;
    }

    const size_t total_outputs = payload.batch_size * payload.sae_dim;
    const size_t required_blocks = (total_outputs + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(required_blocks, static_cast<size_t>(kMaxBlocks)));

    // Launch asynchronously on the model's CUDA stream.
    sae_encode_kernel<<<blocks, kThreadsPerBlock, 0, payload.stream>>>(
        payload.d_input,
        payload.d_weights,
        payload.d_bias,
        payload.d_latents,
        payload.hidden_dim,
        payload.sae_dim,
        total_outputs
    );

    // Catches launch-time errors only; async execution errors surface on the
    // stream (or at the next synchronization point).
    CUDA_CHECK(cudaGetLastError());
}

} // namespace spire
