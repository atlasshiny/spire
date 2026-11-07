#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "[-] No CUDA-capable devices detected or driver error." << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[+] Spire initialized successfully." << std::endl;
    std::cout << "Target GPU: " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ")" << std::endl;
    
    return 0;
}