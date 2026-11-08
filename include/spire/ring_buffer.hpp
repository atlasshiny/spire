#pragma once

#include <cstddef>
#include <queue>
#include <mutex>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <spire/device_buffer.hpp>

namespace spire {

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
