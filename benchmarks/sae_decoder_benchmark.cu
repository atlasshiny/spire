#include <benchmark/benchmark.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <spire/cuda_macros.hpp>
#include <spire/device_buffer.hpp>
#include <spire/kernels.cuh>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

struct Shape {
    std::size_t batch;
    std::size_t hidden;
    std::size_t sae;
};

std::string label(const Shape& shape) {
    return "b=" + std::to_string(shape.batch) +
           " h=" + std::to_string(shape.hidden) +
           " s=" + std::to_string(shape.sae);
}

void fill_bf16(__nv_bfloat16* data, std::size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = __float2bfloat16(dist(rng));
    }
}

void fill_f32(float* data, std::size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

void BM_SaeDecoder(benchmark::State& state, Shape shape) {
    const std::size_t latent_count = shape.batch * shape.sae;
    const std::size_t weight_count = shape.sae * shape.hidden;
    const std::size_t output_count = shape.batch * shape.hidden;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    spire::DeviceBuffer<__nv_bfloat16> d_weights;
    spire::DeviceBuffer<float> d_bias;
    spire::DeviceBuffer<float> d_latents;
    spire::DeviceBuffer<__nv_bfloat16> d_output;
    try {
        std::vector<__nv_bfloat16> h_weights(weight_count);
        fill_bf16(h_weights.data(), weight_count, 0x12345678u);
        std::vector<float> h_bias(shape.hidden);
        fill_f32(h_bias.data(), shape.hidden, 0x9abcdef0u);
        std::vector<float> h_latents(latent_count);
        fill_f32(h_latents.data(), latent_count, 0x0badf00du);

        d_weights = spire::make_device_buffer<__nv_bfloat16>(weight_count);
        d_bias = spire::make_device_buffer<float>(shape.hidden);
        d_latents = spire::make_device_buffer<float>(latent_count);
        d_output = spire::make_device_buffer<__nv_bfloat16>(output_count);

        CUDA_CHECK(cudaMemcpyAsync(d_weights.get(), h_weights.data(),
                                   weight_count * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_bias.get(), h_bias.data(),
                                   shape.hidden * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_latents.get(), h_latents.data(),
                                   latent_count * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    // Warm up on the real stream so context/init costs are not measured.
    spire::launch_sae_decoder(d_latents.get(), d_weights.get(), d_bias.get(),
                              d_output.get(), shape.batch, shape.sae,
                              shape.hidden, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double total_seconds = 0.0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(d_weights.get());
        benchmark::DoNotOptimize(d_latents.get());
        benchmark::DoNotOptimize(d_output.get());

        CUDA_CHECK(cudaEventRecord(start, stream));
        spire::launch_sae_decoder(d_latents.get(), d_weights.get(),
                                  d_bias.get(), d_output.get(), shape.batch,
                                  shape.sae, shape.hidden, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        benchmark::ClobberMemory();

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
        state.SetIterationTime(seconds);
        total_seconds += seconds;
    }

    const double iterations = static_cast<double>(state.iterations());
    const double seconds = total_seconds / iterations;
    const double outputs_per_second =
        static_cast<double>(output_count) / seconds;
    const double flops =
        2.0 * static_cast<double>(shape.batch) *
        static_cast<double>(shape.sae) *
        static_cast<double>(shape.hidden);
    // Logical weight bytes: the kernel reads each decoder column per batch item.
    const double weight_bytes_per_second =
        static_cast<double>(shape.batch * shape.sae * shape.hidden) *
        sizeof(__nv_bfloat16) / seconds;

    state.counters["ms/launch"] = seconds * 1e3;
    state.counters["outputs/s"] = outputs_per_second;
    state.counters["TFLOP/s"] = flops / seconds / 1e12;
    state.counters["weight GB/s"] = weight_bytes_per_second / 1e9;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(output_count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(shape.batch * shape.hidden *
                                                 shape.sae) *
                            static_cast<int64_t>(sizeof(__nv_bfloat16)));
    state.SetLabel(label(shape));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

constexpr Shape kBatch1 = {1, 4096, 16384};
constexpr Shape kBatch2 = {2, 4096, 16384};
constexpr Shape kBatch4 = {4, 4096, 16384};
constexpr Shape kBatch8 = {8, 4096, 16384};
constexpr Shape kBatch16 = {16, 4096, 16384};

constexpr Shape kHidden2048 = {4, 2048, 16384};
constexpr Shape kHidden8192 = {4, 8192, 16384};

constexpr Shape kSae4096 = {4, 4096, 4096};
constexpr Shape kSae32768 = {4, 4096, 32768};
constexpr Shape kSae65536 = {4, 4096, 65536};

} // namespace

BENCHMARK_CAPTURE(BM_SaeDecoder, b1_h4096_s16384, kBatch1)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b2_h4096_s16384, kBatch2)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b4_h4096_s16384, kBatch4)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b8_h4096_s16384, kBatch8)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b16_h4096_s16384, kBatch16)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b4_h2048_s16384, kHidden2048)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b4_h8192_s16384, kHidden8192)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b4_h4096_s4096, kSae4096)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b4_h4096_s32768, kSae32768)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeDecoder, b4_h4096_s65536, kSae65536)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
