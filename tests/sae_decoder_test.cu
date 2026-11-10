#include <spire/cuda_macros.hpp>
#include <spire/device_buffer.hpp>
#include <spire/kernels.cuh>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

struct Shape {
    std::size_t batch;
    std::size_t hidden;
    std::size_t sae;
};

// Runs one decoder configuration end-to-end and compares the BF16 result against
// a host reference: x'[b, h] = sum_k z[b, k] * W_dec[k, h] + b_dec[h].
bool run_case(const Shape& shape, uint32_t seed) {
    const std::size_t latent_count = shape.batch * shape.sae;
    const std::size_t weight_count = shape.sae * shape.hidden;
    const std::size_t output_count = shape.batch * shape.hidden;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<__nv_bfloat16> h_weights(weight_count);
    for (auto& w : h_weights) {
        w = __float2bfloat16(dist(rng));
    }
    std::vector<float> h_bias(shape.hidden);
    for (auto& b : h_bias) {
        b = dist(rng);
    }
    std::vector<float> h_latents(latent_count);
    for (auto& z : h_latents) {
        z = dist(rng);
    }

    spire::DeviceBuffer<__nv_bfloat16> d_weights =
        spire::make_device_buffer<__nv_bfloat16>(weight_count);
    spire::DeviceBuffer<float> d_bias =
        spire::make_device_buffer<float>(shape.hidden);
    spire::DeviceBuffer<float> d_latents =
        spire::make_device_buffer<float>(latent_count);
    spire::DeviceBuffer<__nv_bfloat16> d_output =
        spire::make_device_buffer<__nv_bfloat16>(output_count);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weights.get(), h_weights.data(),
                               weight_count * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bias.get(), h_bias.data(),
                               shape.hidden * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_latents.get(), h_latents.data(),
                               latent_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    spire::launch_sae_decoder(d_latents.get(), d_weights.get(), d_bias.get(),
                              d_output.get(), shape.batch, shape.sae,
                              shape.hidden, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> h_output(output_count);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output.get(),
                          output_count * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamDestroy(stream));

    // Host reference in the same accumulation order as the kernel.
    std::size_t mismatches = 0;
    float max_abs_diff = 0.0f;
    for (std::size_t b = 0; b < shape.batch; ++b) {
        const float* z = h_latents.data() + b * shape.sae;
        for (std::size_t h = 0; h < shape.hidden; ++h) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < shape.sae; ++k) {
                sum += z[k] * __bfloat162float(h_weights[k * shape.hidden + h]);
            }
            const float expected = sum + h_bias[h];
            const float actual = __bfloat162float(h_output[b * shape.hidden + h]);
            const float diff = std::fabs(actual - expected);
            max_abs_diff = std::max(max_abs_diff, diff);
            // BF16 output rounds to ~3 significant digits; allow a relative
            // tolerance plus a small absolute floor for near-zero values.
            if (diff > 1e-2f * std::max(1.0f, std::fabs(expected))) {
                ++mismatches;
            }
        }
    }

    std::printf("  b=%zu h=%zu s=%zu: mismatches=%zu max_abs_diff=%.6f\n",
                shape.batch, shape.hidden, shape.sae, mismatches,
                max_abs_diff);
    return mismatches == 0;
}

} // namespace

int main() {
    std::printf("[*] SAE decoder kernel test\n");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "[-] No CUDA-capable devices detected.\n");
        return 1;
    }

    expect(run_case(Shape{2, 8, 16}, 1u), "even hidden_dim (vectorized path)");
    expect(run_case(Shape{2, 5, 7}, 2u), "odd hidden_dim (scalar fallback)");
    expect(run_case(Shape{1, 4096, 2048}, 3u), "realistic batch-1 shape");
    expect(run_case(Shape{2, 4096, 2048}, 4u), "realistic batch-2 shape");

    if (failures == 0) {
        std::printf("[+] SAE decoder verified against the host reference.\n");
        return 0;
    }
    std::fprintf(stderr, "[-] %d decoder test(s) failed.\n", failures);
    return 1;
}
