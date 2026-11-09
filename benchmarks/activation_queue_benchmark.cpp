#include <benchmark/benchmark.h>

#include <spire/activation_queue.hpp>
#include <spire/legacy/activation_queue_legacy.hpp>

#include <barrier>
#include <cstddef>
#include <cstdint>
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

} // namespace

BENCHMARK(BM_NewActivationQueue)->Iterations(5)->UseRealTime();
BENCHMARK(BM_LegacyActivationQueue)->Iterations(5)->UseRealTime();
BENCHMARK_MAIN();
