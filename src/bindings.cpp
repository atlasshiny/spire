#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <spire/SaeWeights.hpp>
#include <spire/activation_queue.hpp>
#include <spire/activation_payload.hpp>
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

        {
            py::gil_scoped_release release;
            if (!queue_.push(std::move(payload))) {
                throw std::runtime_error("Activation queue is full");
            }
        }
    }
};

PYBIND11_MODULE(spire_cpp, m) {
    m.doc() = "Spire real-time SAE activation interception daemon";

    py::class_<PySpireDaemon>(m, "SpireDaemon")
        .def(py::init<const std::string&>())
        .def("push_activation", &PySpireDaemon::push_activation, 
             py::call_guard<py::gil_scoped_release>(),
             "Push a PyTorch hidden state tensor into the SPSC ring buffer.");
}
