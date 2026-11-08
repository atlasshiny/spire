#include <spire/cuda_macros.hpp>
#include <spire/daemon.hpp>
#include <spire/ring_buffer.hpp>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    std::cout << "[*] Initializing Spire Test Harness...\n";

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "[-] No CUDA-capable devices detected.\n";
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "[+] Target GPU: " << prop.name << "\n";

    constexpr std::size_t batch_size = 2;
    constexpr std::size_t hidden_dim = 4096;
    constexpr std::size_t total_elements = batch_size * hidden_dim;
    constexpr std::size_t bytes = total_elements * sizeof(__nv_bfloat16);

    // Host reference data with distinct values so the round-trip check is meaningful.
    std::vector<__nv_bfloat16> h_input(total_elements);
    for (std::size_t i = 0; i < total_elements; ++i) {
        h_input[i] = __float2bfloat16(static_cast<float>(i % 251) * 0.5f);
    }

    __nv_bfloat16* d_tensor = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tensor), bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tensor, h_input.data(), bytes, cudaMemcpyHostToDevice, stream));

    spire::ActivationQueue queue;
    {
        spire::Daemon daemon(queue);
        daemon.start();

        std::cout << "[*] Pushing activation payload into queue...\n";
        queue.push(spire::ActivationPayload{
            .d_tensor = d_tensor,
            .batch_size = batch_size,
            .hidden_dim = hidden_dim,
            .stream = stream,
        });

        // Give the worker time to pop the payload and enqueue the kernel.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        daemon.stop();
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> h_output(total_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_tensor, bytes, cudaMemcpyDeviceToHost));

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < total_elements; ++i) {
        if (__bfloat162float(h_output[i]) != __bfloat162float(h_input[i])) {
            ++mismatches;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_tensor));

    if (mismatches != 0) {
        std::cerr << "[-] Verification failed: " << mismatches << " / " << total_elements
                  << " elements differ.\n";
        return 1;
    }

    std::cout << "[+] Verified " << total_elements << " bf16 elements in place.\n";
    std::cout << "[+] Spire test harness completed successfully.\n";
    return 0;
}
