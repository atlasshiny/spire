#pragma once

#include <cstddef>
#include <queue>
#include <mutex>
#include <spire/activation_payload.hpp>

namespace spire {

namespace legacy {

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

} // namespace legacy

} // namespace spire
