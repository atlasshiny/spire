#include <spire/daemon.hpp>
#include <iostream>
#include <chrono>

// Forward declaration of our CUDA kernel host launcher
extern "C" void launch_sae_kernel(const spire::ActivationPayload& payload);

namespace spire {

void Daemon::start() {
    running_.store(true, std::memory_order_release);
    
    // Launch the background thread
    worker_thread_ = std::jthread([this](std::stop_token stoken) {
        worker_loop(stoken);
    });
}

void Daemon::stop() {
    running_.store(false, std::memory_order_release);
}

void Daemon::worker_loop(std::stop_token stoken) {
    while (!stoken.stop_requested() && running_.load(std::memory_order_acquire)) {
        ActivationPayload payload;
        
        // Try to grab a payload from the thread-safe queue
        if (queue_.try_pop(payload)) {
            // Dispatch to GPU compute pipeline
            launch_sae_kernel(payload);
        } else {
            // Queue is empty; yield briefly to avoid burning CPU cycles
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
}

} // namespace spire