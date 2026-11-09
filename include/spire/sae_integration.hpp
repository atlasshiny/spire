#pragma once

#include <spire/SaeWeights.hpp>

#include <cstddef>
#include <string>

namespace spire {

// Describes an SAE's encoder parameters so callers can load weights from an
// external library (SAELens, TransformerLens, etc.) without hardcoding tensor
// names at every call site.
//
// SAELens and TransformerLens both expose the encoder as:
//   W_enc: [d_sae, d_in]  (row-major)
//   b_enc: [d_sae]
//
// The C++ side expects weights as [sae_dim, hidden_dim] and bias as [sae_dim],
// so d_sae -> sae_dim and d_in -> hidden_dim.
struct SAEIntegrationConfig {
    std::string weight_name = "W_enc";
    std::string bias_name = "b_enc";
    std::size_t expected_sae_dim = 0;     // d_sae -> sae_dim
    std::size_t expected_hidden_dim = 0;  // d_in -> hidden_dim
};

// Convenience: build a SaeConfig from an SAEIntegrationConfig. The tensor names
// differ (encoder.weight/encoder.bias vs W_enc/b_enc) but the dimensions map 1:1.
inline SaeConfig to_sae_config(const SAEIntegrationConfig& config) {
    return SaeConfig{
        .weight_name = config.weight_name,
        .bias_name = config.bias_name,
        .expected_sae_dim = config.expected_sae_dim,
        .expected_hidden_dim = config.expected_hidden_dim,
    };
}

} // namespace spire
