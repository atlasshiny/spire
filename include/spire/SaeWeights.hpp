#pragma once

#include <cstddef>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <spire/device_buffer.hpp>

namespace spire {

// SAE parameters resident on the GPU, ready to feed into ActivationPayload.
// Weights are bf16 and row-major [sae_dim, hidden_dim]; bias is fp32 [sae_dim].
struct SaeWeights {
    DeviceBuffer<__nv_bfloat16> d_weights;
    DeviceBuffer<float> d_bias;
    std::size_t sae_dim = 0;
    std::size_t hidden_dim = 0;
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
