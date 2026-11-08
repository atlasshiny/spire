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

// Describes how an SAE checkpoint names and lays out its tensors. Keeping this
// in one object lets checkpoints that differ from the default convention load
// without hardcoding names at every call site. An expected dimension of zero
// means "accept whatever the file contains".
struct SaeConfig {
    std::string weight_name = "encoder.weight";
    std::string bias_name = "encoder.bias";
    std::size_t expected_sae_dim = 0;    // [sae_dim] rows of the weight matrix
    std::size_t expected_hidden_dim = 0; // columns of the weight matrix
};

// Loads `config.weight_name` / `config.bias_name` from a .safetensors file.
// Weights must be [sae_dim, hidden_dim] and bias [sae_dim]; both are checked
// against the config when expected dimensions are set. Supported source dtypes:
// BF16, F16, F32 (weights are converted to bf16, bias to fp32). Uploads are
// queued on `stream` and synchronized before returning, so the result is ready
// for immediate use.
SaeWeights load_safetensors(const std::string& filepath,
                            const SaeConfig& config = {},
                            cudaStream_t stream = nullptr);

} // namespace spire
