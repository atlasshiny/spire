#include <spire/dtype_convert.cuh>
#include <spire/cuda_macros.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>

namespace spire {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxBlocks = 65535;

// Elements copied to the GPU per staging round trip. This bounds the extra
// device memory the conversion needs, independent of tensor size.
constexpr std::size_t kStageElements = 1u << 20;

// Bytes per direct host->device copy. Chunking keeps the driver's internal
// pinned staging allocation from growing with the tensor size.
constexpr std::size_t kCopyBytes = 8u << 20;

__device__ __forceinline__ __nv_bfloat16 convert_bf16(float value) {
    return __float2bfloat16(value);
}

__device__ __forceinline__ __nv_bfloat16 convert_bf16(__half value) {
    return __float2bfloat16(__half2float(value));
}

__device__ __forceinline__ float convert_f32(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float convert_f32(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename SrcT>
__global__ void convert_to_bf16_kernel(const SrcT* __restrict__ src,
                                       __nv_bfloat16* __restrict__ dst,
                                       std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        dst[i] = convert_bf16(src[i]);
    }
}

template <typename SrcT>
__global__ void convert_to_f32_kernel(const SrcT* __restrict__ src,
                                      float* __restrict__ dst,
                                      std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        dst[i] = convert_f32(src[i]);
    }
}

// Uploads bytes that need no conversion, in bounded chunks.
template <typename T>
void copy_chunks(const void* host, T* dst, std::size_t count, cudaStream_t stream) {
    const auto* src = static_cast<const std::byte*>(host);
    auto* out = reinterpret_cast<std::byte*>(dst);
    const std::size_t total_bytes = count * sizeof(T);

    for (std::size_t offset = 0; offset < total_bytes; offset += kCopyBytes) {
        const std::size_t chunk = std::min(kCopyBytes, total_bytes - offset);
        CUDA_CHECK(cudaMemcpyAsync(out + offset, src + offset, chunk,
                                   cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

unsigned int blocks_for(std::size_t count) {
    const std::size_t needed = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return static_cast<unsigned int>(std::min<std::size_t>(needed, kMaxBlocks));
}

// Copies `count` source elements to a device staging buffer in chunks and runs
// the conversion kernel on each chunk. Staging instead of pinning the mapped
// file keeps the extra device memory bounded and avoids relying on
// cudaHostRegister support for file-backed mappings.
template <typename SrcT>
void convert_chunks_to_bf16(const void* host, __nv_bfloat16* dst,
                            std::size_t count, cudaStream_t stream) {
    const auto* bytes = static_cast<const std::byte*>(host);
    const DeviceBuffer<SrcT> staging =
        make_device_buffer<SrcT>(std::min(count, kStageElements));

    for (std::size_t offset = 0; offset < count; offset += kStageElements) {
        const std::size_t chunk = std::min(kStageElements, count - offset);
        CUDA_CHECK(cudaMemcpyAsync(staging.get(), bytes + offset * sizeof(SrcT),
                                   chunk * sizeof(SrcT), cudaMemcpyHostToDevice,
                                   stream));
        convert_to_bf16_kernel<SrcT>
            <<<blocks_for(chunk), kThreadsPerBlock, 0, stream>>>(
                staging.get(), dst + offset, chunk);
        CUDA_CHECK(cudaGetLastError());
    }

    // Drain before the staging buffer is released so its lifetime never overlaps
    // a pending copy; the caller can then use `dst` immediately.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename SrcT>
void convert_chunks_to_f32(const void* host, float* dst, std::size_t count,
                           cudaStream_t stream) {
    const auto* bytes = static_cast<const std::byte*>(host);
    const DeviceBuffer<SrcT> staging =
        make_device_buffer<SrcT>(std::min(count, kStageElements));

    for (std::size_t offset = 0; offset < count; offset += kStageElements) {
        const std::size_t chunk = std::min(kStageElements, count - offset);
        CUDA_CHECK(cudaMemcpyAsync(staging.get(), bytes + offset * sizeof(SrcT),
                                   chunk * sizeof(SrcT), cudaMemcpyHostToDevice,
                                   stream));
        convert_to_f32_kernel<SrcT>
            <<<blocks_for(chunk), kThreadsPerBlock, 0, stream>>>(
                staging.get(), dst + offset, chunk);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace

DeviceBuffer<__nv_bfloat16> upload_bf16(const TensorView& view, cudaStream_t stream) {
    const std::size_t count = view.element_count();
    if (count == 0) {
        return DeviceBuffer<__nv_bfloat16>{};
    }

    DeviceBuffer<__nv_bfloat16> dst = make_device_buffer<__nv_bfloat16>(count);
    switch (view.dtype) {
        case DType::BF16:
            // Already bf16: upload the mapped bytes directly.
            copy_chunks(view.data.data(), dst.get(), count, stream);
            break;
        case DType::F16:
            convert_chunks_to_bf16<__half>(view.data.data(), dst.get(), count, stream);
            break;
        case DType::F32:
            convert_chunks_to_bf16<float>(view.data.data(), dst.get(), count, stream);
            break;
    }
    return dst;
}

DeviceBuffer<float> upload_f32(const TensorView& view, cudaStream_t stream) {
    const std::size_t count = view.element_count();
    if (count == 0) {
        return DeviceBuffer<float>{};
    }

    DeviceBuffer<float> dst = make_device_buffer<float>(count);
    switch (view.dtype) {
        case DType::F32:
            // Already fp32: upload the mapped bytes directly.
            copy_chunks(view.data.data(), dst.get(), count, stream);
            break;
        case DType::F16:
            convert_chunks_to_f32<__half>(view.data.data(), dst.get(), count, stream);
            break;
        case DType::BF16:
            convert_chunks_to_f32<__nv_bfloat16>(view.data.data(), dst.get(), count,
                                                 stream);
            break;
    }
    return dst;
}

} // namespace spire
