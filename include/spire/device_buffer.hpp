#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace spire {

// Frees CUDA device memory when the last owner of a buffer goes away.
template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

// Reference-counted RAII handle for device memory. Copying the handle shares
// ownership; the allocation is freed once the last copy is destroyed.
template <typename T>
using DeviceBuffer = std::shared_ptr<T>;

// Allocates `count` elements of device memory. Throws on CUDA failure so the
// caller can decide how to handle it instead of aborting the process.
template <typename T>
DeviceBuffer<T> make_device_buffer(std::size_t count) {
    if (count == 0) {
        return DeviceBuffer<T>{};
    }

    void* raw = nullptr;
    const cudaError_t err = cudaMalloc(&raw, count * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }

    return DeviceBuffer<T>(static_cast<T*>(raw), CudaDeleter<T>{});
}

} // namespace spire
