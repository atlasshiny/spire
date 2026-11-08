#include <spire/SaeWeights.hpp>
#include <spire/SafetensorsFile.hpp>
#include <spire/dtype_convert.cuh>

#include <stdexcept>
#include <string>

namespace spire {

SaeWeights load_safetensors(const std::string& filepath, const SaeConfig& config,
                            cudaStream_t stream) {
    // Parsing is CUDA-free; only the upload below touches the device.
    const SafetensorsFile file(filepath);
    const TensorView& weight = file.tensor(config.weight_name);
    const TensorView& bias = file.tensor(config.bias_name);

    if (weight.shape.size() != 2) {
        throw std::runtime_error("'" + config.weight_name +
                                 "' must be 2-D [sae_dim, hidden_dim]");
    }
    if (bias.shape.size() != 1) {
        throw std::runtime_error("'" + config.bias_name + "' must be 1-D [sae_dim]");
    }

    const std::size_t sae_dim = weight.shape[0];
    const std::size_t hidden_dim = weight.shape[1];
    if (bias.shape[0] != sae_dim) {
        throw std::runtime_error("bias length does not match weight rows");
    }
    if (config.expected_sae_dim != 0 && sae_dim != config.expected_sae_dim) {
        throw std::runtime_error("'" + config.weight_name + "' has sae_dim " +
                                 std::to_string(sae_dim) + ", expected " +
                                 std::to_string(config.expected_sae_dim));
    }
    if (config.expected_hidden_dim != 0 && hidden_dim != config.expected_hidden_dim) {
        throw std::runtime_error("'" + config.weight_name + "' has hidden_dim " +
                                 std::to_string(hidden_dim) + ", expected " +
                                 std::to_string(config.expected_hidden_dim));
    }

    SaeWeights result;
    result.sae_dim = sae_dim;
    result.hidden_dim = hidden_dim;
    // dtype conversion (when needed) happens on the GPU, so no host-side
    // temporary proportional to the weight matrix is allocated.
    result.d_weights = upload_bf16(weight, stream);
    result.d_bias = upload_f32(bias, stream);
    return result;
}

} // namespace spire
