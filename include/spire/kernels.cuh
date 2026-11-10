#pragma once

#include <cstddef>
#include <spire/activation_payload.hpp>

namespace spire {

void launch_sae_kernel(const ActivationPayload& payload);

// SAE decoder projection: output = latents * W_dec + b_dec.
// `latents` is [batch_size, sae_dim] FP32, `weights` is [sae_dim, hidden_dim]
// BF16 row-major, `bias` is [hidden_dim] FP32, and `output` is
// [batch_size, hidden_dim] BF16. Launches asynchronously on `stream`.
void launch_sae_decoder(const float* latents,
                        const __nv_bfloat16* weights,
                        const float* bias,
                        __nv_bfloat16* output,
                        std::size_t batch_size,
                        std::size_t sae_dim,
                        std::size_t hidden_dim,
                        cudaStream_t stream);

} // namespace spire
