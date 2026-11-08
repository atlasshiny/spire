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

// Values are chosen to be exactly representable in both bf16 and fp16 so tests
// can compare converted results bit-for-bit.
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

// Truncating float->fp16 encoder. Only correct for the fixture values, which
// are exactly representable in fp16 (mantissa fits in 10 bits).
inline std::uint16_t to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent < -14 || exponent > 15) {
        return sign;
    }
    return static_cast<std::uint16_t>(
        sign | static_cast<std::uint16_t>((exponent + 15) << 10) |
        static_cast<std::uint16_t>(mantissa >> 13));
}

enum class FixtureDType { F32, F16, BF16 };

inline const char* dtype_name(FixtureDType dtype) {
    switch (dtype) {
        case FixtureDType::F32: return "F32";
        case FixtureDType::F16: return "F16";
        case FixtureDType::BF16: return "BF16";
    }
    return "F32";
}

inline std::vector<std::byte> encode_values(const std::vector<float>& values,
                                            FixtureDType dtype) {
    if (dtype == FixtureDType::F32) {
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }

    std::vector<std::uint16_t> packed(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        packed[i] = dtype == FixtureDType::F16 ? to_f16_bits(values[i])
                                               : to_bf16_bits(values[i]);
    }
    std::vector<std::byte> bytes(packed.size() * sizeof(std::uint16_t));
    std::memcpy(bytes.data(), packed.data(), bytes.size());
    return bytes;
}

inline void write_safetensors(const std::string& path, const Fixture& f,
                              FixtureDType dtype) {
    const std::vector<std::byte> weight_bytes = encode_values(f.weights, dtype);
    const std::vector<std::byte> bias_bytes = encode_values(f.bias, dtype);

    nlohmann::json header;
    header["encoder.weight"] = {
        {"dtype", dtype_name(dtype)},
        {"shape", {f.sae_dim, f.hidden_dim}},
        {"data_offsets", {0, weight_bytes.size()}},
    };
    header["encoder.bias"] = {
        {"dtype", dtype_name(dtype)},
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
