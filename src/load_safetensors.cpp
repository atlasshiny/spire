#include <spire/SaeWeights.hpp>
#include <spire/SafetensorsFile.hpp>
#include <spire/cuda_macros.hpp>

#include <cuda_fp16.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace spire {
namespace {

std::vector<__nv_bfloat16> to_bf16(const TensorView& view) {
    const std::size_t count = view.element_count();
    std::vector<__nv_bfloat16> out(count);

    if (view.dtype == DType::BF16) {
        std::memcpy(out.data(), view.data.data(), view.data.size());
    } else if (view.dtype == DType::F16) {
        std::vector<__half> tmp(count);
        std::memcpy(tmp.data(), view.data.data(), view.data.size());
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = __float2bfloat16(__half2float(tmp[i]));
        }
    } else { // F32
        std::vector<float> tmp(count);
        std::memcpy(tmp.data(), view.data.data(), view.data.size());
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = __float2bfloat16(tmp[i]);
        }
    }

    return out;
}

std::vector<float> to_float(const TensorView& view) {
    const std::size_t count = view.element_count();
    std::vector<float> out(count);

    if (view.dtype == DType::F32) {
        std::memcpy(out.data(), view.data.data(), view.data.size());
    } else if (view.dtype == DType::F16) {
        std::vector<__half> tmp(count);
        std::memcpy(tmp.data(), view.data.data(), view.data.size());
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = __half2float(tmp[i]);
        }
    } else { // BF16
        std::vector<__nv_bfloat16> tmp(count);
        std::memcpy(tmp.data(), view.data.data(), view.data.size());
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = __bfloat162float(tmp[i]);
        }
    }

    return out;
}

} // namespace

SaeWeights load_safetensors(const std::string& filepath, cudaStream_t stream,
                            const std::string& weight_name,
                            const std::string& bias_name) {
    // Parsing is CUDA-free; only the upload below touches the device.
    const SafetensorsFile file(filepath);
    const TensorView& weight = file.tensor(weight_name);
    const TensorView& bias = file.tensor(bias_name);

    if (weight.shape.size() != 2) {
        throw std::runtime_error("'" + weight_name + "' must be 2-D [sae_dim, hidden_dim]");
    }
    if (bias.shape.size() != 1) {
        throw std::runtime_error("'" + bias_name + "' must be 1-D [sae_dim]");
    }

    const std::size_t sae_dim = weight.shape[0];
    const std::size_t hidden_dim = weight.shape[1];
    if (bias.shape[0] != sae_dim) {
        throw std::runtime_error("bias length does not match weight rows");
    }

    const std::vector<__nv_bfloat16> h_weights = to_bf16(weight);
    const std::vector<float> h_bias = to_float(bias);

    SaeWeights result;
    result.sae_dim = sae_dim;
    result.hidden_dim = hidden_dim;
    result.d_weights = to_device<__nv_bfloat16>(h_weights, stream);
    result.d_bias = to_device<float>(h_bias, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

} // namespace spire
