#include <benchmark/benchmark.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <spire/activation_payload.hpp>
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

std::string label(const Shape& s) {
    return "b=" + std::to_string(s.batch) + " h=" + std::to_string(s.hidden) +
           " s=" + std::to_string(s.sae);
}

void fill_bf16(__nv_bfloat16* d, std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<__nv_bfloat16> h(n);
    for (auto& v : h) v = __float2bfloat16(dist(rng));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
}
void fill_f32(float* d, std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> h(n);
    for (auto& v : h) v = dist(rng);
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

using EncFn = void (*)(const spire::ActivationPayload&);
using DecFn = void (*)(const float*, const __nv_bfloat16*, const float*,
                       __nv_bfloat16*, std::size_t, std::size_t, std::size_t,
                       cudaStream_t);

void BM_Encoder(benchmark::State& state, Shape s, EncFn fn, const char* backend) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    spire::ActivationPayload p{};
    p.d_input = spire::make_device_buffer<__nv_bfloat16>(s.batch * s.hidden);
    p.d_weights = spire::make_device_buffer<__nv_bfloat16>(s.sae * s.hidden);
    p.d_bias = spire::make_device_buffer<float>(s.sae);
    p.d_latents = spire::make_device_buffer<float>(s.batch * s.sae);
    p.batch_size = s.batch;
    p.hidden_dim = s.hidden;
    p.sae_dim = s.sae;
    p.stream = stream;
    fill_bf16(p.d_input.get(), s.batch * s.hidden, 1u);
    fill_bf16(p.d_weights.get(), s.sae * s.hidden, 2u);
    fill_f32(p.d_bias.get(), s.sae, 3u);

    fn(p);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double total = 0.0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(p.d_input.get());
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn(p);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        benchmark::ClobberMemory();
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double sec = ms / 1000.0;
        state.SetIterationTime(sec);
        total += sec;
    }

    const double sec = total / static_cast<double>(state.iterations());
    const double flops = 2.0 * s.batch * s.sae * s.hidden;
    state.counters["ms/launch"] = sec * 1e3;
    state.counters["latents/s"] = (s.batch * s.sae) / sec;
    state.counters["TFLOP/s"] = flops / sec / 1e12;
    state.counters["weight GB/s"] =
        (double)(s.batch * s.sae * s.hidden) * sizeof(__nv_bfloat16) / sec / 1e9;
    state.SetLabel(label(s) + " " + backend);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void BM_Decoder(benchmark::State& state, Shape s, DecFn fn, const char* backend) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto z = spire::make_device_buffer<float>(s.batch * s.sae);
    auto w = spire::make_device_buffer<__nv_bfloat16>(s.sae * s.hidden);
    auto bias = spire::make_device_buffer<float>(s.hidden);
    auto out = spire::make_device_buffer<__nv_bfloat16>(s.batch * s.hidden);
    fill_f32(z.get(), s.batch * s.sae, 4u);
    fill_bf16(w.get(), s.sae * s.hidden, 5u);
    fill_f32(bias.get(), s.hidden, 6u);

    fn(z.get(), w.get(), bias.get(), out.get(), s.batch, s.sae, s.hidden, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double total = 0.0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(w.get());
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn(z.get(), w.get(), bias.get(), out.get(), s.batch, s.sae, s.hidden,
           stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        benchmark::ClobberMemory();
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double sec = ms / 1000.0;
        state.SetIterationTime(sec);
        total += sec;
    }

    const double sec = total / static_cast<double>(state.iterations());
    const double flops = 2.0 * s.batch * s.sae * s.hidden;
    state.counters["ms/launch"] = sec * 1e3;
    state.counters["outputs/s"] = (s.batch * s.hidden) / sec;
    state.counters["TFLOP/s"] = flops / sec / 1e12;
    state.counters["weight GB/s"] =
        (double)(s.batch * s.sae * s.hidden) * sizeof(__nv_bfloat16) / sec / 1e9;
    state.SetLabel(label(s) + " " + backend);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

constexpr Shape kB1 = {1, 4096, 16384};
constexpr Shape kB4 = {4, 4096, 16384};
constexpr Shape kB16 = {16, 4096, 16384};
constexpr Shape kB32 = {32, 4096, 16384};

} // namespace

#define ENC_CASES(prefix, fn, name)                                          \
    BENCHMARK_CAPTURE(BM_Encoder, prefix##_b1, kB1, fn, name)                \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);                    \
    BENCHMARK_CAPTURE(BM_Encoder, prefix##_b4, kB4, fn, name)                \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);                    \
    BENCHMARK_CAPTURE(BM_Encoder, prefix##_b16, kB16, fn, name)              \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);                    \
    BENCHMARK_CAPTURE(BM_Encoder, prefix##_b32, kB32, fn, name)              \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);

#define DEC_CASES(prefix, fn, name)                                          \
    BENCHMARK_CAPTURE(BM_Decoder, prefix##_b1, kB1, fn, name)                \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);                    \
    BENCHMARK_CAPTURE(BM_Decoder, prefix##_b4, kB4, fn, name)                \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);                    \
    BENCHMARK_CAPTURE(BM_Decoder, prefix##_b16, kB16, fn, name)              \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);                    \
    BENCHMARK_CAPTURE(BM_Decoder, prefix##_b32, kB32, fn, name)              \
        ->UseManualTime()->Unit(benchmark::kMicrosecond);

ENC_CASES(kernel, &spire::launch_sae_kernel, "kernel")
ENC_CASES(gemm, &spire::launch_sae_kernel_gemm, "gemm")

DEC_CASES(kernel, &spire::launch_sae_decoder, "kernel")
DEC_CASES(gemm, &spire::launch_sae_decoder_gemm, "gemm")

BENCHMARK_MAIN();
