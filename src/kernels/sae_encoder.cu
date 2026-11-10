#include <spire/kernels.cuh>

namespace spire {

void launch_sae_kernel(const ActivationPayload& payload) {
    launch_sae_kernel_gemm(payload);
}

} // namespace spire
