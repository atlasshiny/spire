#include <benchmark/benchmark.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <spire/SaeWeights.hpp>
#include <spire/activation_payload.hpp>
#include <spire/activation_queue.hpp>
#include <spire/cuda_macros.hpp>
#include <spire/device_buffer.hpp>
#include <spire/kernels.cuh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>
namespace {

constexpr std::size_t kMessages = 128;

struct Shape {
    std::size_t batch;
    std::size_t hidden;
    std::size_t sae;
};

using Clock = std::chrono::steady_clock;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
}

std::string label(const Shape& shape, bool with_kernel, std::size_t capacity) {
    return "b=" + std::to_string(shape.batch) +
           " h=" + std::to_string(shape.hidden) +
           " s=" + std::to_string(shape.sae) +
           (with_kernel ? " async+kernel" : " async handoff") +
           " cap=" + std::to_string(capacity);
}

struct LatencyStats {
    int64_t p50_ns;
    int64_t p95_ns;
    int64_t p99_ns;
};

LatencyStats summarize(std::vector<int64_t>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    const auto percentile = [&](double fraction) {
        const auto index = std::min(
            latencies.size() - 1,
            static_cast<std::size_t>(fraction * latencies.size()));
        return latencies[index];
    };
    return {percentile(0.50), percentile(0.95), percentile(0.99)};
}

void BM_Pipeline(benchmark::State& state, Shape shape, bool with_kernel,
                 std::size_t capacity) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    spire::SaeWeights model;
    spire::ActivationQueue queue(capacity);

    try {
        model.d_weights = spire::make_device_buffer<__nv_bfloat16>(
            shape.sae * shape.hidden);
        model.d_bias = spire::make_device_buffer<float>(shape.sae);
        model.sae_dim = shape.sae;
        model.hidden_dim = shape.hidden;

        auto input = spire::make_device_buffer<__nv_bfloat16>(
            shape.batch * shape.hidden);
        auto latents = spire::make_device_buffer<float>(shape.batch * shape.sae);

        CUDA_CHECK(cudaMemsetAsync(input.get(), 0,
                                   shape.batch * shape.hidden *
                                       sizeof(__nv_bfloat16),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(
            model.d_weights.get(), 0,
            shape.sae * shape.hidden * sizeof(__nv_bfloat16), stream));
        CUDA_CHECK(cudaMemsetAsync(model.d_bias.get(), 0,
                                   shape.sae * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(latents.get(), 0,
                                   shape.batch * shape.sae * sizeof(float),
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    // Warm up the pipeline with one full round trip so CUDA context / driver /
    // kernel init costs are not measured.
    {
        spire::ActivationQueue warmup_queue(capacity);
        auto warmup_input = spire::make_device_buffer<__nv_bfloat16>(
            shape.batch * shape.hidden);
        auto warmup_latents = spire::make_device_buffer<float>(
            shape.batch * shape.sae);
        spire::SaeWeights warmup_model = model; // shares device buffers
        spire::ActivationPayload warmup_payload =
            warmup_model.make_payload(std::move(warmup_input),
                                      std::move(warmup_latents),
                                      shape.batch, stream);
        spire::launch_sae_kernel(warmup_payload);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<int64_t> all_latencies;
    all_latencies.reserve(kMessages * state.iterations());
    double total_batch_seconds = 0.0;

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<int64_t> enqueued(kMessages);
        std::vector<int64_t> completed(kMessages);
        std::atomic<bool> go{false};
        std::atomic<std::size_t> completed_count{0};

        std::thread producer([&] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (std::size_t i = 0; i < kMessages; ++i) {
                if (capacity == 1) {
                    // Serialize request/response so this mode measures
                    // per-message latency instead of queue backlog.
                    while (completed_count.load(std::memory_order_acquire) < i) {
                        std::this_thread::yield();
                    }
                }

                auto input = spire::make_device_buffer<__nv_bfloat16>(
                    shape.batch * shape.hidden);
                auto latents = spire::make_device_buffer<float>(
                    shape.batch * shape.sae);

                spire::ActivationPayload payload =
                    model.make_payload(std::move(input), std::move(latents),
                                       shape.batch, stream);

                enqueued[i] = now_ns();
                while (!queue.push(std::move(payload))) {
                    enqueued[i] = now_ns();
                    std::this_thread::yield();
                }
            }
        });

        std::thread consumer([&] {
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            spire::ActivationPayload payload;
            std::size_t i = 0;
            while (i < kMessages) {
                if (!queue.try_pop(payload)) {
                    std::this_thread::yield();
                    continue;
                }

                if (with_kernel) {
                    spire::launch_sae_kernel(payload);
                    CUDA_CHECK(cudaStreamSynchronize(payload.stream));
                }
                completed[i] = now_ns();
                completed_count.store(i + 1, std::memory_order_release);
                ++i;
            }
        });

        const auto start = Clock::now();
        go.store(true, std::memory_order_release);
        producer.join();
        consumer.join();
        const auto end = Clock::now();
        state.ResumeTiming();

        const double batch_seconds =
            std::chrono::duration<double>(end - start).count();
        total_batch_seconds += batch_seconds;
        state.SetIterationTime(batch_seconds);
        std::vector<int64_t> latencies(kMessages);
        for (std::size_t i = 0; i < kMessages; ++i) {
            latencies[i] = completed[i] - enqueued[i];
        }
        all_latencies.insert(all_latencies.end(),
                             latencies.begin(), latencies.end());

        benchmark::ClobberMemory();
    }

    const LatencyStats stats = summarize(all_latencies);
    const auto iterations = static_cast<double>(state.iterations());
    const double average_batch_seconds =
        total_batch_seconds / iterations;

    state.counters["batch ms"] = average_batch_seconds * 1e3;
    state.counters["per-msg us"] =
        average_batch_seconds * 1e6 / static_cast<double>(kMessages);
    state.counters["msg/s"] =
        static_cast<double>(kMessages) / average_batch_seconds;
    state.counters["p50_us"] = static_cast<double>(stats.p50_ns) / 1e3;
    state.counters["p95_us"] = static_cast<double>(stats.p95_ns) / 1e3;
    state.counters["p99_us"] = static_cast<double>(stats.p99_ns) / 1e3;

    state.SetItemsProcessed(static_cast<int64_t>(iterations) *
                            static_cast<int64_t>(kMessages));
    state.SetLabel(label(shape, with_kernel, capacity));
    benchmark::DoNotOptimize(all_latencies.data());

    CUDA_CHECK(cudaStreamDestroy(stream));
}

constexpr Shape kBatch1 = {1, 4096, 16384};
constexpr Shape kBatch4 = {4, 4096, 16384};
constexpr Shape kBatch16 = {16, 4096, 16384};
constexpr Shape kHidden8192 = {4, 8192, 16384};

} // namespace

BENCHMARK_CAPTURE(BM_Pipeline, b1_h4096_s16384_latency, kBatch1, true, 1)
    ->UseManualTime()->Iterations(3)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Pipeline, b4_h4096_s16384_latency, kBatch4, true, 1)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Pipeline, b16_h4096_s16384_latency, kBatch16, true, 1)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Pipeline, b4_h8192_s16384_latency, kHidden8192, true, 1)
    ->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Pipeline, b4_h4096_s16384_throughput, kBatch4, true, 32)
    ->UseManualTime()->Iterations(3)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Pipeline, b4_h4096_s16384_handoff, kBatch4, false, 32)
    ->UseManualTime()->Iterations(5)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
