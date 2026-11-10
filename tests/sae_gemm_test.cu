#include <spire/activation_payload.hpp>
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

bool run_encoder(const Shape& shape, uint32_t seed) {
    const std::size_t input_count = shape.batch * shape.hidden;
    const std::size_t weight_count = shape.sae * shape.hidden;
    const std::size_t latent_count = shape.batch * shape.sae;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<__nv_bfloat16> h_x(input_count);
    for (auto& v : h_x) v = __float2bfloat16(dist(rng));
    std::vector<__nv_bfloat16> h_w(weight_count);
    for (auto& v : h_w) v = __float2bfloat16(dist(rng));
    std::vector<float> h_bias(shape.sae);
    for (auto& v : h_bias) v = dist(rng);

    spire::ActivationPayload payload{};
    payload.d_input = spire::make_device_buffer<__nv_bfloat16>(input_count);
    payload.d_weights = spire::make_device_buffer<__nv_bfloat16>(weight_count);
    payload.d_bias = spire::make_device_buffer<float>(shape.sae);
    payload.d_latents = spire::make_device_buffer<float>(latent_count);
    payload.batch_size = shape.batch;
    payload.hidden_dim = shape.hidden;
    payload.sae_dim = shape.sae;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    payload.stream = stream;
    CUDA_CHECK(cudaMemcpyAsync(payload.d_input.get(), h_x.data(),
                               input_count * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(payload.d_weights.get(), h_w.data(),
                               weight_count * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(payload.d_bias.get(), h_bias.data(),
                               shape.sae * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    spire::launch_sae_kernel_gemm(payload);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_out(latent_count);
    CUDA_CHECK(cudaMemcpy(h_out.data(), payload.d_latents.get(),
                          latent_count * sizeof(float), cudaMemcpyDeviceToHost));

    std::size_t mismatches = 0;
    float max_abs_diff = 0.0f;
    for (std::size_t b = 0; b < shape.batch; ++b) {
        for (std::size_t s = 0; s < shape.sae; ++s) {
            float sum = 0.0f;
            for (std::size_t h = 0; h < shape.hidden; ++h) {
                sum += __bfloat162float(h_x[b * shape.hidden + h]) *
                       __bfloat162float(h_w[s * shape.hidden + h]);
            }
            const float expected = std::fmax(0.0f, sum + h_bias[s]);
            const float actual = h_out[b * shape.sae + s];
            const float diff = std::fabs(actual - expected);
            max_abs_diff = std::fmax(max_abs_diff, diff);
            if (diff > 1e-2f * std::fmax(1.0f, std::fabs(expected))) {
                ++mismatches;
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("  encoder b=%zu h=%zu s=%zu: mismatches=%zu max_abs_diff=%.6f\n",
                shape.batch, shape.hidden, shape.sae, mismatches, max_abs_diff);
    return mismatches == 0;
}

bool run_decoder(const Shape& shape, uint32_t seed) {
    const std::size_t latent_count = shape.batch * shape.sae;
    const std::size_t weight_count = shape.sae * shape.hidden;
    const std::size_t output_count = shape.batch * shape.hidden;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<__nv_bfloat16> h_w(weight_count);
    for (auto& v : h_w) v = __float2bfloat16(dist(rng));
    std::vector<float> h_bias(shape.hidden);
    for (auto& v : h_bias) v = dist(rng);
    std::vector<float> h_z(latent_count);
    for (auto& v : h_z) v = dist(rng);

    spire::DeviceBuffer<__nv_bfloat16> d_w =
        spire::make_device_buffer<__nv_bfloat16>(weight_count);
    spire::DeviceBuffer<float> d_bias =
        spire::make_device_buffer<float>(shape.hidden);
    spire::DeviceBuffer<float> d_z =
        spire::make_device_buffer<float>(latent_count);
    spire::DeviceBuffer<__nv_bfloat16> d_out =
        spire::make_device_buffer<__nv_bfloat16>(output_count);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w.get(), h_w.data(),
                               weight_count * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bias.get(), h_bias.data(),
                               shape.hidden * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z.get(), h_z.data(),
                               latent_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    spire::launch_sae_decoder_gemm(d_z.get(), d_w.get(), d_bias.get(),
                                   d_out.get(), shape.batch, shape.sae,
                                   shape.hidden, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> h_out(output_count);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(),
                          output_count * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    std::size_t mismatches = 0;
    float max_abs_diff = 0.0f;
    for (std::size_t b = 0; b < shape.batch; ++b) {
        for (std::size_t h = 0; h < shape.hidden; ++h) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < shape.sae; ++k) {
                sum += h_z[b * shape.sae + k] *
                       __bfloat162float(h_w[k * shape.hidden + h]);
            }
            const float expected = sum + h_bias[h];
            const float actual = __bfloat162float(h_out[b * shape.hidden + h]);
            const float diff = std::fabs(actual - expected);
            max_abs_diff = std::fmax(max_abs_diff, diff);
            if (diff > 1e-2f * std::fmax(1.0f, std::fabs(expected))) {
                ++mismatches;
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("  decoder b=%zu h=%zu s=%zu: mismatches=%zu max_abs_diff=%.6f\n",
                shape.batch, shape.hidden, shape.sae, mismatches, max_abs_diff);
    return mismatches == 0;
}

} // namespace

int main() {
    std::printf("[*] SAE cuBLAS GEMM kernel test\n");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "[-] No CUDA-capable devices detected.\n");
        return 1;
    }

    expect(run_encoder(Shape{2, 8, 16}, 11u), "encoder GEMM small");
    expect(run_encoder(Shape{2, 5, 7}, 12u), "encoder GEMM odd dims");
    expect(run_encoder(Shape{1, 4096, 2048}, 13u), "encoder GEMM realistic");
    expect(run_encoder(Shape{4, 4096, 2048}, 14u), "encoder GEMM batch");

    expect(run_decoder(Shape{2, 8, 16}, 21u), "decoder GEMM small");
    expect(run_decoder(Shape{2, 5, 7}, 22u), "decoder GEMM odd dims");
    expect(run_decoder(Shape{1, 4096, 2048}, 23u), "decoder GEMM realistic");
    expect(run_decoder(Shape{4, 4096, 2048}, 24u), "decoder GEMM batch");

    if (failures == 0) {
        std::printf("[+] cuBLAS GEMM kernels verified against the host reference.\n");
        return 0;
    }
    std::fprintf(stderr, "[-] %d GEMM test(s) failed.\n", failures);
    return 1;
}
