#include <benchmark/benchmark.h>

#include <spire/activation_queue.hpp>
#include <spire/legacy/activation_queue_legacy.hpp>

#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace {

constexpr std::size_t message_count = 100'000;
constexpr std::size_t queue_capacity = 1024;

spire::ActivationPayload make_payload(std::size_t value) {
    spire::ActivationPayload payload{};
    payload.batch_size = value;
    return payload;
}

template <typename Queue>
void run_spsc(benchmark::State& state, Queue& queue, bool bounded) {
    for (auto _ : state) {
        std::barrier start(3);

        std::thread producer([&] {
            start.arrive_and_wait();
            for (std::size_t value = 0; value < message_count; ++value) {
                auto payload = make_payload(value);
                if constexpr (std::is_same_v<Queue, spire::ActivationQueue>) {
                    while (!queue.push(std::move(payload))) {
                        std::this_thread::yield();
                    }
                } else {
                    queue.push(payload);
                }
            }
        });

        std::thread consumer([&] {
            start.arrive_and_wait();
            spire::ActivationPayload payload{};
            for (std::size_t value = 0; value < message_count; ++value) {
                while (!queue.try_pop(payload)) {
                    std::this_thread::yield();
                }
                benchmark::DoNotOptimize(payload.batch_size);
            }
        });

        state.PauseTiming();
        start.arrive_and_wait();
        state.ResumeTiming();
        producer.join();
        consumer.join();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(message_count) * 2);
    state.SetLabel(bounded ? "bounded SPSC" : "mutex queue");
}

void BM_NewActivationQueue(benchmark::State& state) {
    spire::ActivationQueue queue(queue_capacity);
    run_spsc(state, queue, true);
}

void BM_LegacyActivationQueue(benchmark::State& state) {
    spire::legacy::ActivationQueue queue;
    run_spsc(state, queue, false);
}

void BM_ActivationQueueCapacity(benchmark::State& state) {
    spire::ActivationQueue queue(static_cast<std::size_t>(state.range(0)));
    run_spsc(state, queue, true);
    state.SetLabel("capacity " + std::to_string(state.range(0)));
}

void BM_QueuePushOnly(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto queue = std::make_unique<spire::ActivationQueue>(
            message_count + 1);
        state.ResumeTiming();

        for (std::size_t value = 0; value < message_count; ++value) {
            queue->push(make_payload(value));
        }

        state.PauseTiming();
        queue.reset();
        state.ResumeTiming();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(message_count));
    state.SetLabel("producer only");
}

void BM_QueuePopOnly(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        auto queue = std::make_unique<spire::ActivationQueue>(
            message_count + 1);
        for (std::size_t value = 0; value < message_count; ++value) {
            queue->push(make_payload(value));
        }
        spire::ActivationPayload payload{};
        state.ResumeTiming();

        for (std::size_t value = 0; value < message_count; ++value) {
            while (!queue->try_pop(payload)) {
                std::this_thread::yield();
            }
            benchmark::DoNotOptimize(payload.batch_size);
        }

        state.PauseTiming();
        queue.reset();
        state.ResumeTiming();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(message_count));
    state.SetLabel("consumer only");
}

void BM_QueuePushFull(benchmark::State& state) {
    constexpr std::size_t capacity = 64;
    spire::ActivationQueue queue(capacity);
    auto payload = make_payload(0);
    for (std::size_t i = 0; i < capacity; ++i) {
        queue.push(payload);
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(queue.push(payload));
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetLabel("push into full queue");
}

void BM_QueuePopEmpty(benchmark::State& state) {
    spire::ActivationQueue queue(queue_capacity);
    spire::ActivationPayload payload{};

    for (auto _ : state) {
        benchmark::DoNotOptimize(queue.try_pop(payload));
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetLabel("pop from empty queue");
}

} // namespace

BENCHMARK(BM_NewActivationQueue)->Iterations(5)->UseRealTime();
BENCHMARK(BM_LegacyActivationQueue)->Iterations(5)->UseRealTime();

BENCHMARK(BM_ActivationQueueCapacity)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Iterations(3)
    ->UseRealTime();

BENCHMARK(BM_QueuePushOnly)->Iterations(5)->UseRealTime();
BENCHMARK(BM_QueuePopOnly)->Iterations(5)->UseRealTime();
BENCHMARK(BM_QueuePushFull)->Iterations(5);
BENCHMARK(BM_QueuePopEmpty)->Iterations(5);

BENCHMARK_MAIN();
