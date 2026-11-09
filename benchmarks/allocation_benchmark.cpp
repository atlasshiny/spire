#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <spire/cuda_macros.hpp>
#include <spire/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr std::size_t kPoolSize = 8;

void BM_CudaMalloc(benchmark::State& state) {
    const auto bytes = static_cast<std::size_t>(state.range(0));

    // Warm up the CUDA context before the timed loop.
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    CUDA_CHECK(cudaFree(ptr));

    for (auto _ : state) {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        benchmark::DoNotOptimize(ptr);
        CUDA_CHECK(cudaFree(ptr));
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(bytes));
    state.SetLabel("cudaMalloc + cudaFree");
}

void BM_DeviceBuffer(benchmark::State& state) {
    const auto bytes = static_cast<std::size_t>(state.range(0));
    const auto count = bytes / sizeof(float);

    {
        auto warmup = spire::make_device_buffer<float>(count);
        benchmark::DoNotOptimize(warmup);
    }

    for (auto _ : state) {
        auto buffer = spire::make_device_buffer<float>(count);
        benchmark::DoNotOptimize(buffer);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(bytes));
    state.SetLabel("DeviceBuffer (allocate + free)");
}

void BM_CudaMallocAsync(benchmark::State& state) {
    const auto bytes = static_cast<std::size_t>(state.range(0));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Keep freed blocks in the pool so this measures the pool fast path rather
    // than returning memory to the driver every iteration.
    cudaMemPool_t pool = nullptr;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
    std::uint64_t release_threshold = UINT64_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));

    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocAsync(&ptr, bytes, stream));
    CUDA_CHECK(cudaFreeAsync(ptr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        CUDA_CHECK(cudaMallocAsync(&ptr, bytes, stream));
        benchmark::DoNotOptimize(ptr);
        CUDA_CHECK(cudaFreeAsync(ptr, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(bytes));
    state.SetLabel("cudaMallocAsync (pooled)");

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void BM_PooledReuse(benchmark::State& state) {
    const auto bytes = static_cast<std::size_t>(state.range(0));
    const auto count = bytes / sizeof(float);

    std::vector<spire::DeviceBuffer<float>> pool;
    pool.reserve(kPoolSize);
    for (std::size_t i = 0; i < kPoolSize; ++i) {
        pool.push_back(spire::make_device_buffer<float>(count));
    }

    std::size_t next = 0;
    for (auto _ : state) {
        auto& buffer = pool[next++ % kPoolSize];
        benchmark::DoNotOptimize(buffer.get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(bytes));
    state.SetLabel("preallocated pool (no allocation)");
}

void BM_DeviceBufferBatch(benchmark::State& state) {
    const auto bytes = static_cast<std::size_t>(state.range(0));
    const auto count = bytes / sizeof(float);

    {
        std::vector<spire::DeviceBuffer<float>> warmup;
        warmup.reserve(kPoolSize);
        for (std::size_t i = 0; i < kPoolSize; ++i) {
            warmup.push_back(spire::make_device_buffer<float>(count));
        }
        benchmark::DoNotOptimize(warmup);
    }

    for (auto _ : state) {
        std::vector<spire::DeviceBuffer<float>> buffers;
        buffers.reserve(kPoolSize);
        for (std::size_t i = 0; i < kPoolSize; ++i) {
            buffers.push_back(spire::make_device_buffer<float>(count));
        }
        benchmark::DoNotOptimize(buffers);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kPoolSize));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kPoolSize) *
                            static_cast<int64_t>(bytes));
    state.SetLabel("DeviceBuffer batch of 8");
}

} // namespace

BENCHMARK(BM_CudaMalloc)->Arg(1 << 20)->Arg(16 << 20)->Arg(64 << 20);
BENCHMARK(BM_DeviceBuffer)->Arg(1 << 20)->Arg(16 << 20)->Arg(64 << 20);
BENCHMARK(BM_CudaMallocAsync)->Arg(1 << 20)->Arg(16 << 20)->Arg(64 << 20);
BENCHMARK(BM_PooledReuse)->Arg(1 << 20)->Arg(16 << 20)->Arg(64 << 20);
BENCHMARK(BM_DeviceBufferBatch)->Arg(1 << 20)->Arg(16 << 20)->Arg(64 << 20);

BENCHMARK_MAIN();
