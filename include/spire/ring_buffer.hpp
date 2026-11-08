#pragma once

#include <queue>
#include <mutex>
#include <optional>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace spire {

struct ActivationPayload {
    __nv_bfloat16* d_tensor;
    size_t batch_size;
    size_t hidden_dim;
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