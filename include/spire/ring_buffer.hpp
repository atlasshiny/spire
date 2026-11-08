#pragma once

#include <cstddef>
#include <memory>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

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

// One unit of work for the GPU pipeline: an input activation block plus the
// SAE parameters and the destination buffer for the resulting latents.
// The buffers are reference-counted, so a payload can be copied into the queue
// while the producer keeps reading its own handle (e.g. the latents).
struct ActivationPayload {
    DeviceBuffer<__nv_bfloat16> d_input;   // [batch_size, hidden_dim]
    DeviceBuffer<__nv_bfloat16> d_weights; // [sae_dim, hidden_dim], row-major, bf16
    DeviceBuffer<float> d_bias;            // [sae_dim]
    DeviceBuffer<float> d_latents;         // [batch_size, sae_dim]
    size_t batch_size;
    size_t hidden_dim;
    size_t sae_dim;
    cudaStream_t stream;
};

class ActivationQueue {
private:
    std::queue<ActivationPayload> queue_;
    mutable std::mutex mutex_;

public:
    ActivationQueue() = default;
    ~ActivationQueue() = default;

    // Delete copy/move constructors to prevent accidental thread synchronization bugs
    ActivationQueue(const ActivationQueue&) = delete;
    ActivationQueue& operator=(const ActivationQueue&) = delete;

    void push(const ActivationPayload& payload) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(payload);
    }

    bool try_pop(ActivationPayload& payload) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        payload = queue_.front();
        queue_.pop();
        return true;
    }
};

} // namespace spire
