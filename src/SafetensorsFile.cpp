#include <spire/SafetensorsFile.hpp>

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace spire {
namespace {

using json = nlohmann::json;

constexpr std::uint64_t kMaxHeaderBytes = 100ull * 1024ull * 1024ull;

DType parse_dtype(const std::string& dtype) {
    if (dtype == "F32") {
        return DType::F32;
    }
    if (dtype == "F16") {
        return DType::F16;
    }
    if (dtype == "BF16") {
        return DType::BF16;
    }
    throw std::runtime_error("unsupported safetensors dtype '" + dtype +
                             "' (supported: BF16, F16, F32)");
}

std::size_t checked_element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (const std::size_t dim : shape) {
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("tensor shape overflows size_t");
        }
        count *= dim;
    }
    return count;
}

void validate_tensor_layout(const std::string& name, const json& tensor,
                            std::size_t begin, std::size_t end,
                            std::size_t payload_size) {
    if (end < begin || end > payload_size) {
        throw std::runtime_error("tensor '" + name + "' data_offsets exceed the data section");
    }

    const DType dtype = parse_dtype(tensor.at("dtype").get<std::string>());
    const auto shape = tensor.at("shape").get<std::vector<std::size_t>>();
    const std::size_t count = checked_element_count(shape);
    const std::size_t elem_size = dtype_size(dtype);

    if (count > std::numeric_limits<std::size_t>::max() / elem_size) {
        throw std::runtime_error("tensor '" + name + "' byte size overflows size_t");
    }
    if (end - begin != count * elem_size) {
        throw std::runtime_error("tensor '" + name + "' byte length does not match its shape");
    }
}

} // namespace

std::size_t TensorView::element_count() const {
    return checked_element_count(shape);
}

SafetensorsFile::SafetensorsFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open safetensors file: " + path);
    }

    file.seekg(0, std::ios::end);
    const std::streamoff size_off = file.tellg();
    if (size_off < 0) {
        throw std::runtime_error("failed to determine size of '" + path + "'");
    }
    const std::uint64_t file_size = static_cast<std::uint64_t>(size_off);
    if (file_size < sizeof(std::uint64_t)) {
        throw std::runtime_error("'" + path + "' is not a valid safetensors file");
    }

    file.seekg(0, std::ios::beg);
    buffer_.resize(static_cast<std::size_t>(file_size));
    file.read(reinterpret_cast<char*>(buffer_.data()),
              static_cast<std::streamsize>(buffer_.size()));
    if (!file) {
        throw std::runtime_error("failed to read '" + path + "'");
    }

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, buffer_.data(), sizeof(header_len));
    if (header_len > kMaxHeaderBytes) {
        throw std::runtime_error("safetensors header is unreasonably large");
    }
    if (header_len > file_size - sizeof(header_len)) {
        throw std::runtime_error("safetensors header length exceeds file size");
    }

    const auto* header_begin =
        reinterpret_cast<const char*>(buffer_.data() + sizeof(header_len));
    json header;
    try {
        header = json::parse(header_begin, header_begin + header_len);
    } catch (const json::exception& e) {
        throw std::runtime_error(std::string("invalid safetensors JSON header: ") + e.what());
    }
    if (!header.is_object()) {
        throw std::runtime_error("safetensors header must be a JSON object");
    }

    const std::uint64_t data_start = sizeof(header_len) + header_len;
    const std::size_t payload_size = static_cast<std::size_t>(file_size - data_start);

    for (auto it = header.begin(); it != header.end(); ++it) {
        const std::string& name = it.key();
        if (name == "__metadata__") {
            continue;
        }

        const json& tensor = it.value();
        if (!tensor.is_object() || !tensor.contains("dtype") ||
            !tensor.contains("shape") || !tensor.contains("data_offsets")) {
            throw std::runtime_error("tensor '" + name +
                                     "' is missing dtype/shape/data_offsets");
        }

        const auto offsets = tensor.at("data_offsets").get<std::vector<std::size_t>>();
        if (offsets.size() != 2) {
            throw std::runtime_error("tensor '" + name + "' has invalid data_offsets");
        }

        validate_tensor_layout(name, tensor, offsets[0], offsets[1], payload_size);

        TensorView view;
        view.data = std::span<const std::byte>(
            buffer_.data() + data_start + offsets[0], offsets[1] - offsets[0]);
        view.dtype = parse_dtype(tensor.at("dtype").get<std::string>());
        view.shape = tensor.at("shape").get<std::vector<std::size_t>>();

        tensors_.emplace(name, std::move(view));
    }
}

bool SafetensorsFile::contains(std::string_view name) const {
    return tensors_.find(std::string(name)) != tensors_.end();
}

const TensorView& SafetensorsFile::tensor(std::string_view name) const {
    const auto it = tensors_.find(std::string(name));
    if (it == tensors_.end()) {
        throw std::runtime_error("safetensors file is missing tensor '" + std::string(name) + "'");
    }
    return it->second;
}

} // namespace spire
