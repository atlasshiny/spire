#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spire {

enum class DType { F32, F16, BF16 };

constexpr std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16:
        case DType::BF16: return 2;
    }
    return 0;
}

// A view of one tensor's raw bytes inside the mapped file. The view stays valid
// for the lifetime of the owning SafetensorsFile.
struct TensorView {
    std::span<const std::byte> data;
    DType dtype;
    std::vector<std::size_t> shape;

    std::size_t element_count() const;
};

// Read-only access to a .safetensors file. This layer knows nothing about CUDA:
// it validates the file/header and exposes raw tensor views, so parsing can be
// tested on any host.
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path);
    ~SafetensorsFile() = default;

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;
    SafetensorsFile(SafetensorsFile&&) noexcept = default;
    SafetensorsFile& operator=(SafetensorsFile&&) noexcept = default;

    bool contains(std::string_view name) const;
    const TensorView& tensor(std::string_view name) const;

private:
    std::vector<std::byte> buffer_; // whole file, header included
    std::unordered_map<std::string, TensorView> tensors_;
};

} // namespace spire
