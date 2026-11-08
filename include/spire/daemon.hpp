#pragma once

#include <spire/ring_buffer.hpp>
#include <thread>
#include <atomic>

namespace spire {

class Daemon {
private:
    ActivationQueue& queue_;
    std::atomic<bool> running_{false};
    std::jthread worker_thread_;

    void worker_loop(std::stop_token stoken);

public:
    explicit Daemon(ActivationQueue& queue);
    ~Daemon();

    // Delete copy/move
    Daemon(const Daemon&) = delete;
    Daemon& operator=(const Daemon&) = delete;

    void start();
    void stop();
};

} // namespace spire