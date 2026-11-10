#include <spire/kernels.cuh>

namespace spire {

void launch_sae_decoder(const float* latents,
                        const __nv_bfloat16* weights,
                        const float* bias,
                        __nv_bfloat16* output,
                        std::size_t batch_size,
                        std::size_t sae_dim,
                        std::size_t hidden_dim,
                        cudaStream_t stream) {
    launch_sae_decoder_gemm(latents, weights, bias, output, batch_size,
                            sae_dim, hidden_dim, stream);
}

} // namespace spire
