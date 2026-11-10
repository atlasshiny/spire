#include <spire/kernels.cuh>
#include <spire/sae_gemm_common.hpp>
namespace spire {
void launch_sae_decoder_gemm(const float* latents, const __nv_bfloat16* weights, const float* bias, __nv_bfloat16* output, std::size_t batch_size, std::size_t sae_dim, std::size_t hidden_dim, cudaStream_t stream) {
    if (!latents || !weights || !bias || !output || batch_size == 0) return;
    auto* z = static_cast<__nv_bfloat16*>(thread_scratch(batch_size * sae_dim * sizeof(__nv_bfloat16)));
    f32_to_bf16_kernel<<<blocks_for(batch_size * sae_dim), kThreadsPerBlock, 0, stream>>>(latents, z, batch_size * sae_dim);
    cublasHandle_t handle = thread_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_dim, batch_size, sae_dim, &alpha, weights, CUDA_R_16BF, hidden_dim, z, CUDA_R_16BF, sae_dim, &beta, output, CUDA_R_16BF, hidden_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    bias_add_bf16_kernel<<<blocks_for(batch_size * hidden_dim), kThreadsPerBlock, 0, stream>>>(output, bias, batch_size * hidden_dim, hidden_dim);
}
void launch_sae_decoder_auto(const float* latents, const __nv_bfloat16* weights, const float* bias, __nv_bfloat16* output, std::size_t batch_size, std::size_t sae_dim, std::size_t hidden_dim, cudaStream_t stream) {
    if (use_gemm(batch_size, kDecoderGemmMinBatch)) launch_sae_decoder_gemm(latents, weights, bias, output, batch_size, sae_dim, hidden_dim, stream); else legacy::launch_sae_decoder(latents, weights, bias, output, batch_size, sae_dim, hidden_dim, stream);
}
} // namespace spire
