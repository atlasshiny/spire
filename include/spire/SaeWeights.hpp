#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <spire/device_buffer.hpp>
#include <spire/ring_buffer.hpp>

namespace spire {

// SAE parameters resident on the GPU, ready to feed into ActivationPayload.
// Weights are bf16 and row-major [sae_dim, hidden_dim]; bias is fp32 [sae_dim].
struct SaeWeights {
    DeviceBuffer<__nv_bfloat16> d_weights;
    DeviceBuffer<float> d_bias;
    std::size_t sae_dim = 0;
    std::size_t hidden_dim = 0;

    // Builds a payload that shares the weight/bias buffers (refcount bump, no
    // data copy) and fills in the model dimensions, so callers cannot mismatch
    // sae_dim/hidden_dim or the buffer fields.
    ActivationPayload make_payload(DeviceBuffer<__nv_bfloat16> input,
                                   DeviceBuffer<float> latents,
                                   std::size_t batch_size,
                                   cudaStream_t stream) const {
        return ActivationPayload{
            .d_input = std::move(input),
            .d_weights = d_weights,
            .d_bias = d_bias,
            .d_latents = std::move(latents),
            .batch_size = batch_size,
            .hidden_dim = hidden_dim,
            .sae_dim = sae_dim,
            .stream = stream,
        };
    }
};

// Loads encoder.weight / encoder.bias from a .safetensors file.
// Supported tensor dtypes: BF16, F16, F32 (weights are converted to bf16,
// bias to fp32). The upload is ordered on `stream` and synchronized before
// returning, so the tensors are ready for immediate use.
SaeWeights load_safetensors(const std::string& filepath,
                            cudaStream_t stream = nullptr,
                            const std::string& weight_name = "encoder.weight",
                            const std::string& bias_name = "encoder.bias");

} // namespace spire
