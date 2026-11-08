#include <spire/mapped_file.hpp>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace spire {
namespace {

std::runtime_error errno_error(const char* what, const std::string& path) {
    return std::runtime_error(std::string(what) + " '" + path + "': " +
                              std::strerror(errno));
}

} // namespace

MappedFile::MappedFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw errno_error("failed to open", path);
    }

    struct stat info {};
    if (::fstat(fd, &info) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        throw errno_error("failed to stat", path);
    }
    if (info.st_size < 0) {
        ::close(fd);
        throw std::runtime_error("'" + path + "' has a negative file size");
    }

    size_ = static_cast<std::size_t>(info.st_size);
    if (size_ == 0) {
        // mmap rejects zero-length mappings; an empty file simply has no data.
        ::close(fd);
        return;
    }

    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    const int saved = errno;
    // The mapping keeps the underlying file alive even after the descriptor is
    // closed, so there is no reason to hold it open.
    ::close(fd);
    if (mapped == MAP_FAILED) {
        errno = saved;
        size_ = 0;
        throw errno_error("failed to mmap", path);
    }

    data_ = static_cast<const std::byte*>(mapped);
}

MappedFile::~MappedFile() {
    reset();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void MappedFile::reset() noexcept {
    if (data_ != nullptr) {
        ::munmap(const_cast<std::byte*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }
}

} // namespace spire
