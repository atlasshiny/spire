add_executable(activation_queue_benchmark activation_queue_benchmark.cpp)
target_link_libraries(activation_queue_benchmark
    PRIVATE spire_core CUDA::cudart benchmark::benchmark)
