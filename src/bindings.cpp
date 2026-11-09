#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <spire/SaeWeights.hpp>
#include <spire/activation_queue.hpp>
#include <spire/activation_payload.hpp>
#include <spire/cuda_macros.hpp>
#include <spire/kernels.cuh>

#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace py = pybind11;

class PySpireDaemon {
private:
    // Bound the completed-result buffer so a producer that never polls cannot
    // grow memory without limit. The oldest latents are dropped first.
    static constexpr std::size_t kMaxCompleted = 1024;

    spire::SaeWeights weights_;
    spire::ActivationQueue queue_;

    // Declared before daemon_thread_ so the worker is joined before these are
    // destroyed (members are destroyed in reverse declaration order).
    std::mutex results_mutex_;
    std::deque<torch::Tensor> completed_;
    std::atomic<bool> running_{true};
    std::jthread daemon_thread_;
    c10::DeviceIndex device_index_ = 0;

    void validate_activation(const torch::Tensor& hidden_states) const {
        TORCH_CHECK(hidden_states.is_cuda(),
                    "Activation tensor must be on CUDA VRAM.");
        TORCH_CHECK(hidden_states.device().index() == device_index_,
                    "Activation tensor must be on the same CUDA device as the "
                    "loaded weights.");
        TORCH_CHECK(hidden_states.dim() == 2,
                    "Activation tensor must be [batch_size, hidden_dim].");
        TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16,
                    "Activation tensor must be bfloat16.");
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "Activation tensor must be contiguous.");
        TORCH_CHECK(hidden_states.size(1) ==
                        static_cast<long>(weights_.hidden_dim),
                    "Activation hidden_dim (", hidden_states.size(1),
                    ") does not match the loaded weights (", weights_.hidden_dim,
                    ").");
    }

    torch::Tensor wrap_latents(spire::DeviceBuffer<float> latents,
                               std::size_t batch_size) const {
        return torch::from_blob(
            latents.get(),
            {static_cast<long>(batch_size),
             static_cast<long>(weights_.sae_dim)},
            // The tensor shares ownership of the device allocation.
            [latents](void*) {},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCUDA, device_index_));
    }

public:
    explicit PySpireDaemon(const std::string& safetensors_path) {
        weights_ = spire::load_safetensors(safetensors_path);
        device_index_ = c10::cuda::current_device();

        daemon_thread_ = std::jthread([this](std::stop_token stoken) {
            c10::cuda::CUDAGuard guard(device_index_);

            while (!stoken.stop_requested() &&
                   running_.load(std::memory_order_acquire)) {
                spire::ActivationPayload payload;
                if (!queue_.try_pop(payload)) {
                    std::this_thread::yield();
                    continue;
                }

                spire::launch_sae_kernel(payload);
                // Synchronize before publishing the result so consumers never
                // observe latents that are still being written.
                CUDA_CHECK(cudaStreamSynchronize(payload.stream));

                torch::Tensor latents = wrap_latents(
                    payload.d_latents, payload.batch_size);

                {
                    std::lock_guard<std::mutex> lock(results_mutex_);
                    if (completed_.size() >= kMaxCompleted) {
                        completed_.pop_front();
                    }
                    completed_.push_back(std::move(latents));
                }
            }
        });
    }

    ~PySpireDaemon() {
        running_.store(false, std::memory_order_release);
        daemon_thread_.request_stop();
    }

    void push_activation(torch::Tensor hidden_states) {
        validate_activation(hidden_states);
        c10::cuda::CUDAGuard guard(device_index_);

        const auto batch_size =
            static_cast<std::size_t>(hidden_states.size(0));

        // Non-owning view of the caller's tensor. Capturing the tensor in the
        // deleter keeps its storage alive until the daemon has consumed the
        // payload and the kernel has finished.
        spire::DeviceBuffer<__nv_bfloat16> input_buf(
            reinterpret_cast<__nv_bfloat16*>(hidden_states.data_ptr()),
            [hidden_states](__nv_bfloat16*) {});

        spire::DeviceBuffer<float> latents =
            spire::make_device_buffer<float>(batch_size * weights_.sae_dim);

        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();

        spire::ActivationPayload payload = weights_.make_payload(
            std::move(input_buf), std::move(latents), batch_size, stream);

        // The binding's call_guard already released the GIL for this call, so
        // do not release it a second time here.
        if (!queue_.push(std::move(payload))) {
            throw std::runtime_error("Activation queue is full");
        }
    }

    // Returns the oldest completed latents tensor, or None if none are ready.
    std::optional<torch::Tensor> try_pop_latents() {
        std::lock_guard<std::mutex> lock(results_mutex_);
        if (completed_.empty()) {
            return std::nullopt;
        }

        torch::Tensor latents = std::move(completed_.front());
        completed_.pop_front();
        return latents;
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
        validate_activation(hidden_states);
        c10::cuda::CUDAGuard guard(device_index_);

        const auto batch_size =
            static_cast<std::size_t>(hidden_states.size(0));

        // Non-owning view of the caller's tensor; torch keeps it alive for the
        // duration of this synchronous call.
        spire::DeviceBuffer<__nv_bfloat16> input_buf(
            reinterpret_cast<__nv_bfloat16*>(hidden_states.data_ptr()),
            [](__nv_bfloat16*) { /* owned by torch */ });

        spire::DeviceBuffer<float> latents =
            spire::make_device_buffer<float>(batch_size * weights_.sae_dim);

        const cudaStream_t stream =
            c10::cuda::getCurrentCUDAStream(device_index_).stream();

        spire::ActivationPayload payload = weights_.make_payload(
            std::move(input_buf), latents, batch_size, stream);

        spire::launch_sae_kernel(payload);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return wrap_latents(std::move(latents), batch_size);
    }
};

PYBIND11_MODULE(spire_cpp, m) {
    m.doc() = "Spire real-time SAE activation interception daemon";

    py::class_<PySpireDaemon>(m, "SpireDaemon")
        .def(py::init<const std::string&>())
        .def("push_activation", &PySpireDaemon::push_activation,
             py::call_guard<py::gil_scoped_release>(),
             "Push a PyTorch hidden state tensor into the SPSC ring buffer.")
        .def("try_pop_latents", &PySpireDaemon::try_pop_latents,
             "Return the oldest completed latents tensor, or None if none are "
             "ready.")
        .def("process_sync", &PySpireDaemon::process_sync,
             py::arg("hidden_states"),
             "TESTING ONLY: run the SAE encoder synchronously and return the "
             "latents tensor. Blocks the caller and bypasses the async queue; "
             "use push_activation in production.");
}
