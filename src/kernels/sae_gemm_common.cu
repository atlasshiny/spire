#include <spire/kernels.cuh>
#include <spire/sae_gemm_common.hpp>

namespace spire {
__global__ void bias_relu_f32_kernel(float* __restrict__ out, const float* __restrict__ bias, std::size_t total, std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        out[i] = fmaxf(0.0f, out[i] + bias[i % cols]);
    }
}
__global__ void bias_add_bf16_kernel(__nv_bfloat16* __restrict__ out, const float* __restrict__ bias, std::size_t total, std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        out[i] = __float2bfloat16(__bfloat162float(out[i]) + bias[i % cols]);
    }
}
__global__ void f32_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        dst[i] = __float2bfloat16(src[i]);
    }
}
} // namespace spire
