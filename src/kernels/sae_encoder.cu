#include <spire/kernels.cuh>
#include <spire/cuda_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

namespace {

// Placeholder: performs an in-place identity transform for now.
__global__ void sae_forward_kernel(__nv_bfloat16* __restrict__ data, size_t size) {
    // Grid-stride loop so any tensor size is handled without overflowing the 32-bit grid/thread index math.
    for (size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
         idx < size;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        data[idx] = data[idx];
    }
}

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;

} // namespace

namespace spire {

void launch_sae_kernel(const ActivationPayload& payload) {
    if (payload.d_tensor == nullptr || payload.batch_size == 0 || payload.hidden_dim == 0) {
        return;
    }

    const size_t total_elements = payload.batch_size * payload.hidden_dim;
    const size_t required_blocks = (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const unsigned int blocks =
        static_cast<unsigned int>(std::min(required_blocks, static_cast<size_t>(kMaxBlocks)));

    // Launch asynchronously on the model's CUDA stream.
    sae_forward_kernel<<<blocks, kThreadsPerBlock, 0, payload.stream>>>(
        payload.d_tensor,
        total_elements
    );

    // Catches launch-time errors only; async execution errors surface on the stream (or at the next synchronization point).
    CUDA_CHECK(cudaGetLastError());
}

} // namespace spire
