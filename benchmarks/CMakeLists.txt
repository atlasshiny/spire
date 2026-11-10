add_executable(activation_queue_benchmark activation_queue_benchmark.cpp)
target_link_libraries(activation_queue_benchmark
    PRIVATE spire_core CUDA::cudart benchmark::benchmark)

add_executable(sae_kernel_benchmark sae_kernel_benchmark.cu)
target_link_libraries(sae_kernel_benchmark
    PRIVATE spire_core CUDA::cudart benchmark::benchmark)

add_executable(sae_decoder_benchmark sae_decoder_benchmark.cu)
target_link_libraries(sae_decoder_benchmark
    PRIVATE spire_core CUDA::cudart benchmark::benchmark)

add_executable(sae_gemm_benchmark sae_gemm_benchmark.cu)
target_link_libraries(sae_gemm_benchmark
    PRIVATE spire_core CUDA::cudart CUDA::cublas benchmark::benchmark)

add_executable(pipeline_benchmark pipeline_benchmark.cu)
target_link_libraries(pipeline_benchmark
    PRIVATE spire_core CUDA::cudart benchmark::benchmark)

add_executable(allocation_benchmark allocation_benchmark.cpp)
target_link_libraries(allocation_benchmark
    PRIVATE spire_core CUDA::cudart benchmark::benchmark)
