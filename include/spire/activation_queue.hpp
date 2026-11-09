#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <spire/activation_payload.hpp>

namespace spire {

// A bounded, non-blocking queue for exactly one producer and one consumer.
class ActivationQueue {
public:
    static constexpr std::size_t default_capacity = 1024;

    explicit ActivationQueue(std::size_t capacity = default_capacity)
        : slots_(capacity == 0 ? nullptr : std::make_unique<Slot[]>(capacity)),
          capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("ActivationQueue capacity must be greater than zero");
        }
    }

    ~ActivationQueue() = default;

    ActivationQueue(const ActivationQueue&) = delete;
    ActivationQueue& operator=(const ActivationQueue&) = delete;
    ActivationQueue(ActivationQueue&&) = delete;
    ActivationQueue& operator=(ActivationQueue&&) = delete;

    // Returns false when the queue is full. Only the producer may call this.
    bool push(const ActivationPayload& payload) {
        return emplace(payload);
    }

    // Returns false when the queue is full. Only the producer may call this.
    bool push(ActivationPayload&& payload) {
        return emplace(std::move(payload));
    }

    // Returns false when the queue is empty. Only the consumer may call this.
    bool try_pop(ActivationPayload& payload) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        if (tail == head) {
            return false;
        }

        Slot& slot = slots_[tail % capacity_];
        payload = std::move(*slot.payload);
        slot.payload.reset();
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    std::size_t capacity() const noexcept {
        return capacity_;
    }

    bool empty() const noexcept {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    bool full() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return head - tail == capacity_;
    }

private:
    struct Slot {
        std::optional<ActivationPayload> payload;
    };

    template <typename T>
    bool emplace(T&& payload) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail == capacity_) {
            return false;
        }

        slots_[head % capacity_].payload.emplace(std::forward<T>(payload));
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    std::unique_ptr<Slot[]> slots_;
    const std::size_t capacity_;
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

} // namespace spire
