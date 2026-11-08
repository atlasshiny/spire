#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace spire::test {

struct Fixture {
    std::size_t sae_dim = 4;
    std::size_t hidden_dim = 5;
    std::vector<float> weights; // [sae_dim, hidden_dim]
    std::vector<float> bias;    // [sae_dim]
};

// Values are chosen to be exactly representable in bf16 so tests can compare
// converted results bit-for-bit.
inline Fixture make_fixture() {
    Fixture f;
    f.weights.resize(f.sae_dim * f.hidden_dim);
    for (std::size_t i = 0; i < f.weights.size(); ++i) {
        f.weights[i] = static_cast<float>(i % 17) * 0.25f - 2.0f;
    }
    f.bias.resize(f.sae_dim);
    for (std::size_t j = 0; j < f.bias.size(); ++j) {
        f.bias[j] = 0.5f - 0.125f * static_cast<float>(j);
    }
    return f;
}

inline std::uint16_t to_bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

inline float bf16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

inline void write_safetensors(const std::string& path, const Fixture& f, bool bf16) {
    std::vector<std::byte> weight_bytes;
    std::vector<std::byte> bias_bytes;

    if (bf16) {
        std::vector<std::uint16_t> w(f.weights.size());
        for (std::size_t i = 0; i < f.weights.size(); ++i) w[i] = to_bf16_bits(f.weights[i]);
        std::vector<std::uint16_t> b(f.bias.size());
        for (std::size_t i = 0; i < f.bias.size(); ++i) b[i] = to_bf16_bits(f.bias[i]);

        weight_bytes.resize(w.size() * sizeof(std::uint16_t));
        std::memcpy(weight_bytes.data(), w.data(), weight_bytes.size());
        bias_bytes.resize(b.size() * sizeof(std::uint16_t));
        std::memcpy(bias_bytes.data(), b.data(), bias_bytes.size());
    } else {
        weight_bytes.resize(f.weights.size() * sizeof(float));
        std::memcpy(weight_bytes.data(), f.weights.data(), weight_bytes.size());
        bias_bytes.resize(f.bias.size() * sizeof(float));
        std::memcpy(bias_bytes.data(), f.bias.data(), bias_bytes.size());
    }

    nlohmann::json header;
    header["encoder.weight"] = {
        {"dtype", bf16 ? "BF16" : "F32"},
        {"shape", {f.sae_dim, f.hidden_dim}},
        {"data_offsets", {0, weight_bytes.size()}},
    };
    header["encoder.bias"] = {
        {"dtype", bf16 ? "BF16" : "F32"},
        {"shape", {f.sae_dim}},
        {"data_offsets", {weight_bytes.size(), weight_bytes.size() + bias_bytes.size()}},
    };

    const std::string json = header.dump();
    const std::uint64_t header_len = json.size();

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(json.data(), static_cast<std::streamsize>(json.size()));
    out.write(reinterpret_cast<const char*>(weight_bytes.data()),
              static_cast<std::streamsize>(weight_bytes.size()));
    out.write(reinterpret_cast<const char*>(bias_bytes.data()),
              static_cast<std::streamsize>(bias_bytes.size()));
}

inline void write_truncated(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    const std::uint32_t garbage = 4;
    out.write(reinterpret_cast<const char*>(&garbage), sizeof(garbage));
}

} // namespace spire::test
