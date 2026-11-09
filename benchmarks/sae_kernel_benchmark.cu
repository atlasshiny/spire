#include <benchmark/benchmark.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <spire/SaeWeights.hpp>
#include <spire/activation_payload.hpp>
#include <spire/cuda_macros.hpp>
#include <spire/device_buffer.hpp>
#include <spire/kernels.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

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

// Fills `count` BF16 elements with pseudo-random values so the kernel cannot
// be optimized to a constant-zero memory operation. Uses a simple LCG (no host
// RNG dependency inside the timed loop).
void fill_bf16(__nv_bfloat16* data, std::size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = __float2bfloat16(dist(rng));
    }
}

void BM_SaeKernel(benchmark::State& state, Shape shape) {
    const std::size_t input_count = shape.batch * shape.hidden;
    const std::size_t weight_count = shape.sae * shape.hidden;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    spire::SaeWeights model;
    spire::ActivationPayload payload;
    try {
        std::vector<__nv_bfloat16> h_weights(weight_count);
        fill_bf16(h_weights.data(), weight_count, 0x12345678u);

        model.d_weights = spire::make_device_buffer<__nv_bfloat16>(weight_count);
        model.d_bias = spire::make_device_buffer<float>(shape.sae);
        model.sae_dim = shape.sae;
        model.hidden_dim = shape.hidden;

        auto input = spire::make_device_buffer<__nv_bfloat16>(input_count);
        auto latents = spire::make_device_buffer<float>(shape.batch * shape.sae);

        CUDA_CHECK(cudaMemcpyAsync(model.d_weights.get(), h_weights.data(),
                                   weight_count * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(model.d_bias.get(), 0,
                                   shape.sae * sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        payload = model.make_payload(std::move(input), std::move(latents),
                                     shape.batch, stream);
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    // Warm up the kernel on the real stream so context/init costs are not
    // measured.
    spire::launch_sae_kernel(payload);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double total_seconds = 0.0;
    for (auto _ : state) {
        // Touch input/latents so they cannot be optimized away across
        // iterations; DoNotOptimize on the raw pointers prevents the compiler
        // from treating them as loop-invariant.
        benchmark::DoNotOptimize(payload.d_input.get());
        benchmark::DoNotOptimize(payload.d_latents.get());

        CUDA_CHECK(cudaEventRecord(start, stream));
        spire::launch_sae_kernel(payload);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Barrier so elapsed-time measurement and the kernel are not reordered
        // / elided by the compiler.
        benchmark::ClobberMemory();

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
        state.SetIterationTime(seconds);
        total_seconds += seconds;
    }

    const double iterations = static_cast<double>(state.iterations());
    const double seconds = total_seconds / iterations;
    const double latents_per_second =
        static_cast<double>(shape.batch * shape.sae) / seconds;
    const double flops =
        2.0 * static_cast<double>(shape.batch) *
        static_cast<double>(shape.sae) *
        static_cast<double>(shape.hidden);
    // Logical weight bytes: the kernel reads the full weight row per batch item.
    const double weight_bytes_per_second =
        static_cast<double>(shape.batch * shape.sae * shape.hidden) *
        sizeof(__nv_bfloat16) / seconds;

    state.counters["ms/launch"] = seconds * 1e3;
    state.counters["latents/s"] = latents_per_second;
    state.counters["TFLOP/s"] = flops / seconds / 1e12;
    state.counters["weight GB/s"] = weight_bytes_per_second / 1e9;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(shape.batch * shape.sae));
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

BENCHMARK_CAPTURE(BM_SaeKernel, b1_h4096_s16384, kBatch1)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b2_h4096_s16384, kBatch2)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b4_h4096_s16384, kBatch4)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b8_h4096_s16384, kBatch8)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b16_h4096_s16384, kBatch16)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b4_h2048_s16384, kHidden2048)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b4_h8192_s16384, kHidden8192)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b4_h4096_s4096, kSae4096)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b4_h4096_s32768, kSae32768)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_SaeKernel, b4_h4096_s65536, kSae65536)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
