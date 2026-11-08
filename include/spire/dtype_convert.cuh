#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <spire/SafetensorsFile.hpp>
#include <spire/device_buffer.hpp>

namespace spire {

// Uploads a tensor into fresh device memory. BF16/F32 sources are copied
// verbatim; F16 sources (and cross-dtype requests) are converted by a GPU
// kernel reading from a bounded staging buffer, so no host-side conversion
// buffer for the whole tensor is allocated. Data is queued on `stream`; both
// helpers synchronize the stream before returning, so the buffers are ready for
// use immediately.
DeviceBuffer<__nv_bfloat16> upload_bf16(const TensorView& view, cudaStream_t stream);
DeviceBuffer<float> upload_f32(const TensorView& view, cudaStream_t stream);

} // namespace spire
