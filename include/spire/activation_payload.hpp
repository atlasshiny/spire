#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <spire/device_buffer.hpp>

namespace spire {

struct ActivationPayload {
    DeviceBuffer<__nv_bfloat16> d_input;   // [batch_size, hidden_dim]
    DeviceBuffer<__nv_bfloat16> d_weights; // [sae_dim, hidden_dim], row-major, bf16
    DeviceBuffer<float> d_bias;            // [sae_dim]
    DeviceBuffer<float> d_latents;         // [batch_size, sae_dim]
    size_t batch_size;
    size_t hidden_dim;
    size_t sae_dim;
    cudaStream_t stream;        // daemon's side stream for kernel launch
    cudaEvent_t ready_event;    // recorded on model stream before push
};

} // namespace spire
