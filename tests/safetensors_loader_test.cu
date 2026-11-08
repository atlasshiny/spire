#include "safetensors_fixture.hpp"

#include <spire/SaeWeights.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void check_loaded(const std::string& path, const spire::test::Fixture& fixture,
                  const char* label) {
    const spire::SaeWeights model = spire::load_safetensors(path);

    expect(model.sae_dim == fixture.sae_dim, "loaded sae_dim");
    expect(model.hidden_dim == fixture.hidden_dim, "loaded hidden_dim");

    const std::size_t weight_count = fixture.weights.size();
    std::vector<std::uint16_t> raw_weights(weight_count);
    expect(cudaMemcpy(raw_weights.data(), model.d_weights.get(),
                      weight_count * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess,
           "copy weights back");

    std::vector<float> host_bias(fixture.bias.size());
    expect(cudaMemcpy(host_bias.data(), model.d_bias.get(),
                      host_bias.size() * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess,
           "copy bias back");

    bool weights_match = true;
    for (std::size_t i = 0; i < weight_count; ++i) {
        const float expected = fixture.weights[i];
        const float actual = spire::test::bf16_bits_to_float(raw_weights[i]);
        if (actual != expected) {
            weights_match = false;
            std::fprintf(stderr, "%s: weight[%zu] expected %.6f got %.6f\n",
                         label, i, expected, actual);
            break;
        }
    }
    expect(weights_match, "weights match bf16 reference");
    expect(host_bias == fixture.bias, "bias matches fp32 reference");

    // make_payload must share the weight/bias handles (refcount bump, no copy)
    // and fill the dimensions from the model.
    const spire::ActivationPayload payload =
        model.make_payload(spire::DeviceBuffer<__nv_bfloat16>{},
                           spire::DeviceBuffer<float>{}, 2, nullptr);
    expect(payload.d_weights == model.d_weights, "payload should share weights");
    expect(payload.d_bias == model.d_bias, "payload should share bias");
    expect(payload.sae_dim == model.sae_dim, "payload sae_dim should match model");
    expect(payload.hidden_dim == model.hidden_dim, "payload hidden_dim should match model");
    expect(payload.batch_size == 2, "payload batch_size should be forwarded");
    expect(payload.stream == nullptr, "payload stream should be forwarded");
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    using namespace spire::test;

    const Fixture fixture = make_fixture();

    const fs::path f32_path = fs::temp_directory_path() / "spire_loader_f32.safetensors";
    const fs::path bf16_path = fs::temp_directory_path() / "spire_loader_bf16.safetensors";

    write_safetensors(f32_path.string(), fixture, /*bf16=*/false);
    write_safetensors(bf16_path.string(), fixture, /*bf16=*/true);

    check_loaded(f32_path.string(), fixture, "F32");
    check_loaded(bf16_path.string(), fixture, "BF16");

    bool missing_threw = false;
    try {
        (void)spire::load_safetensors(f32_path.string(), nullptr, "missing.weight");
    } catch (const std::exception&) {
        missing_threw = true;
    }
    expect(missing_threw, "loading a missing tensor should throw");

    fs::remove(f32_path);
    fs::remove(bf16_path);

    if (failures == 0) {
        std::printf("safetensors loader tests passed\n");
        return 0;
    }
    return 1;
}
