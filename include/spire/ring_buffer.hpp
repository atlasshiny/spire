#pragma once

#include <cstddef>
#include <queue>
#include <mutex>
#include <optional>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace spire {

struct ActivationPayload {
    const __nv_bfloat16* d_input; // [batch_size, hidden_dim]
    const float* d_weights; // [sae_dim, hidden_dim], row-major
    const float* d_bias; // [sae_dim]
    float* d_latents; // [batch_size, sae_dim]
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
