#include <spire/kernels.cuh>
#include <spire/sae_gemm_common.hpp>
namespace spire {
void launch_sae_kernel_gemm(const ActivationPayload& payload) {
    if (!payload.d_input || !payload.d_weights || !payload.d_bias || !payload.d_latents || payload.batch_size == 0) return;
    const int batch = static_cast<int>(payload.batch_size), hidden = static_cast<int>(payload.hidden_dim), sae = static_cast<int>(payload.sae_dim);
    cublasHandle_t handle = thread_handle();
    CUBLAS_CHECK(cublasSetStream(handle, payload.stream));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, sae, batch, hidden, &alpha, payload.d_weights.get(), CUDA_R_16BF, hidden, payload.d_input.get(), CUDA_R_16BF, hidden, &beta, payload.d_latents.get(), CUDA_R_32F, sae, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    bias_relu_f32_kernel<<<blocks_for(payload.batch_size * payload.sae_dim), kThreadsPerBlock, 0, payload.stream>>> (payload.d_latents.get(), payload.d_bias.get(), payload.batch_size * payload.sae_dim, payload.sae_dim);
}
void launch_sae_kernel_auto(const ActivationPayload& payload) {
    if (use_gemm(payload.batch_size, kEncoderGemmMinBatch)) launch_sae_kernel_gemm(payload); else legacy::launch_sae_kernel(payload);
}
} // namespace spire
