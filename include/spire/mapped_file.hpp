#pragma once

#include <cstddef>
#include <string>

namespace spire {

// Read-only memory mapping of a whole file. Accessing the data pages it in from
// the kernel page cache on demand, so the file is never copied into a heap
// buffer. The file must not be modified or truncated while the mapping is alive
// (doing so can raise SIGBUS on access), which is fine for immutable
// checkpoints such as .safetensors.
class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    const std::byte* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    void reset() noexcept;

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace spire
