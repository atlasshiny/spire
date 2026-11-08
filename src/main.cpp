#include <spire/cuda_macros.hpp>
#include <spire/daemon.hpp>
#include <spire/ring_buffer.hpp>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
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
    constexpr std::size_t sae_dim = 2048;

    // Synthetic SAE parameters and input activations (deterministic seed).
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> weight_dist(-0.02f, 0.02f);
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);

    std::vector<__nv_bfloat16> h_input(batch_size * hidden_dim);
    for (auto& value : h_input) {
        value = __float2bfloat16(input_dist(rng));
    }

    std::vector<float> h_weights(sae_dim * hidden_dim);
    for (auto& weight : h_weights) {
        weight = weight_dist(rng);
    }

    std::vector<float> h_bias(sae_dim);
    for (auto& bias : h_bias) {
        bias = weight_dist(rng) * 0.1f;
    }

    const std::size_t input_bytes = h_input.size() * sizeof(__nv_bfloat16);
    const std::size_t weights_bytes = h_weights.size() * sizeof(float);
    const std::size_t bias_bytes = h_bias.size() * sizeof(float);
    const std::size_t latents_bytes = batch_size * sae_dim * sizeof(float);

    __nv_bfloat16* d_input = nullptr;
    float* d_weights = nullptr;
    float* d_bias = nullptr;
    float* d_latents = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weights), weights_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bias), bias_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_latents), latents_bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(), input_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weights, h_weights.data(), weights_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bias, h_bias.data(), bias_bytes,
                               cudaMemcpyHostToDevice, stream));

    spire::ActivationQueue queue;
    {
        spire::Daemon daemon(queue);
        daemon.start();

        std::cout << "[*] Pushing activation payload into queue...\n";
        queue.push(spire::ActivationPayload{
            .d_input = d_input,
            .d_weights = d_weights,
            .d_bias = d_bias,
            .d_latents = d_latents,
            .batch_size = batch_size,
            .hidden_dim = hidden_dim,
            .sae_dim = sae_dim,
            .stream = stream,
        });

        // Give the worker time to pop the payload and enqueue the kernel.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        daemon.stop();
    } // Daemon destructor joins the worker before resources are released.

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_latents(batch_size * sae_dim);
    CUDA_CHECK(cudaMemcpyAsync(h_latents.data(), d_latents, latents_bytes,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Host reference: z = ReLU(W_enc * x + b_enc).
    std::vector<float> h_input_float(batch_size * hidden_dim);
    for (std::size_t i = 0; i < h_input.size(); ++i) {
        h_input_float[i] = __bfloat162float(h_input[i]);
    }

    std::size_t mismatches = 0;
    std::size_t nonzero_latents = 0;
    float max_abs_diff = 0.0f;

    for (std::size_t b = 0; b < batch_size; ++b) {
        const float* x = h_input_float.data() + b * hidden_dim;
        for (std::size_t j = 0; j < sae_dim; ++j) {
            const float* w_row = h_weights.data() + j * hidden_dim;

            float sum = h_bias[j];
            for (std::size_t i = 0; i < hidden_dim; ++i) {
                sum += w_row[i] * x[i];
            }

            const float expected = std::max(0.0f, sum);
            const float actual = h_latents[b * sae_dim + j];
            const float diff = std::fabs(actual - expected);
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff > 1e-3f * std::max(1.0f, std::fabs(expected))) {
                ++mismatches;
            }
            if (actual > 0.0f) {
                ++nonzero_latents;
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_latents));

    const std::size_t total_latents = batch_size * sae_dim;
    std::cout << "[*] Latents: " << total_latents
              << ", non-zero: " << nonzero_latents
              << ", max abs diff: " << max_abs_diff << "\n";

    if (mismatches != 0) {
        std::cerr << "[-] Verification failed: " << mismatches << " / " << total_latents
                  << " latents differ.\n";
        return 1;
    }

    std::cout << "[+] Verified " << total_latents << " SAE latents against the host reference.\n";
    std::cout << "[+] Spire test harness completed successfully.\n";
    return 0;
}
