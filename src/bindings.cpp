#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <spire/SaeWeights.hpp>
#include <spire/activation_queue.hpp>
#include <spire/activation_payload.hpp>
#include <spire/cuda_macros.hpp>
#include <spire/kernels.cuh>
#include <thread>
#include <atomic>

namespace py = pybind11;

class PySpireDaemon {
private:
    spire::SaeWeights weights_;
    spire::ActivationQueue queue_;
    std::jthread daemon_thread_;
    std::atomic<bool> running_{true};

public:
    PySpireDaemon(const std::string& safetensors_path) {
        // Load weights using the current Spire API
        weights_ = spire::load_safetensors(safetensors_path);

        // Launch background consumer daemon
        daemon_thread_ = std::jthread([this](std::stop_token stoken) {
            while (!stoken.stop_requested() && running_) {
                spire::ActivationPayload payload;
                if (queue_.try_pop(payload)) {
                    // Note: launch_sae_encoder is the internal kernel harness
                    // In a real scenario, you'd call your math kernels here
                    // For now, we keep the logic matching your original draft
                    // but using the corrected payload members.
                    
                    // Example of how it would be called with current Payload:
                    // launch_sae_encoder(payload.d_input.get(), ...);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    ~PySpireDaemon() {
        running_ = false;
    }

    void push_activation(torch::Tensor hidden_states) {
        TORCH_CHECK(hidden_states.is_cuda(), "Activation tensor must be on CUDA VRAM.");
        
        size_t batch_size = hidden_states.size(0);
        size_t hidden_dim = hidden_states.size(1);

        // Create a payload using SaeWeights::make_payload to ensure 
        // weights/bias are correctly shared and dimensions match.
        
        // 1. Wrap the torch tensor as a DeviceBuffer
        // (Assuming we just want to pass the pointer for now)
        spire::DeviceBuffer<__nv_bfloat16> input_buf(
            reinterpret_cast<__nv_bfloat16*>(hidden_states.data_ptr()),
            []( __nv_bfloat16* ) { /* Torch manages this memory */ }
        );

        // 2. Allocate output latents buffer
        spire::DeviceBuffer<float> latents = spire::make_device_buffer<float>(
            batch_size * weights_.sae_dim
        );

        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        
        spire::ActivationPayload payload = weights_.make_payload(
            std::move(input_buf),
            std::move(latents),
            batch_size,
            stream
        );

        // The binding's call_guard already released the GIL for this call, so
        // do not release it a second time here.
        if (!queue_.push(std::move(payload))) {
            throw std::runtime_error("Activation queue is full");
        }
    }

    // =====================================================================
    // TESTING ONLY.
    //
    // Runs the SAE encoder synchronously on the current stream and returns
    // the latents so tests can compare against a host/PyTorch reference.
    // This blocks the calling thread and bypasses the asynchronous queue,
    // so it must never be used on a production path. Use push_activation
    // for normal operation.
    // =====================================================================
    torch::Tensor process_sync(torch::Tensor hidden_states) {
        TORCH_CHECK(hidden_states.is_cuda(),
                    "Activation tensor must be on CUDA VRAM.");
        TORCH_CHECK(hidden_states.dim() == 2,
                    "Activation tensor must be [batch_size, hidden_dim].");
        TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16,
                    "Activation tensor must be bfloat16.");
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "Activation tensor must be contiguous.");

        const auto batch_size = static_cast<std::size_t>(hidden_states.size(0));
        const auto hidden_dim = static_cast<std::size_t>(hidden_states.size(1));
        TORCH_CHECK(hidden_dim == weights_.hidden_dim,
                    "Activation hidden_dim (", hidden_dim,
                    ") does not match the loaded weights (", weights_.hidden_dim,
                    ").");

        // Non-owning view of the caller's tensor; torch keeps it alive for the
        // duration of this synchronous call.
        spire::DeviceBuffer<__nv_bfloat16> input_buf(
            reinterpret_cast<__nv_bfloat16*>(hidden_states.data_ptr()),
            [](__nv_bfloat16*) { /* owned by torch */ });

        spire::DeviceBuffer<float> latents =
            spire::make_device_buffer<float>(batch_size * weights_.sae_dim);

        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        spire::ActivationPayload payload = weights_.make_payload(
            std::move(input_buf), latents, batch_size, stream);

        spire::launch_sae_kernel(payload);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Return a tensor that shares the latents allocation; the deleter keeps
        // the DeviceBuffer alive until the tensor is destroyed.
        return torch::from_blob(
            latents.get(),
            {static_cast<long>(batch_size), static_cast<long>(weights_.sae_dim)},
            [latents](void*) { /* shared_ptr releases the CUDA allocation */ },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }
};

PYBIND11_MODULE(spire_cpp, m) {
    m.doc() = "Spire real-time SAE activation interception daemon";

    py::class_<PySpireDaemon>(m, "SpireDaemon")
        .def(py::init<const std::string&>())
        .def("push_activation", &PySpireDaemon::push_activation, 
             py::call_guard<py::gil_scoped_release>(),
             "Push a PyTorch hidden state tensor into the SPSC ring buffer.")
        .def("process_sync", &PySpireDaemon::process_sync,
             py::arg("hidden_states"),
             "TESTING ONLY: run the SAE encoder synchronously and return the "
             "latents tensor. Blocks the caller and bypasses the async queue; "
             "use push_activation in production.");
}
