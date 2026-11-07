#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSUCCESS) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)