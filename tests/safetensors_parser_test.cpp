#include "safetensors_fixture.hpp"

#include <spire/SafetensorsFile.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    using namespace spire::test;

    const fs::path fixture_path = fs::temp_directory_path() / "spire_parser_test.safetensors";
    const fs::path truncated_path = fs::temp_directory_path() / "spire_parser_truncated.safetensors";

    const Fixture fixture = make_fixture();
    write_safetensors(fixture_path.string(), fixture, FixtureDType::F32);

    const spire::SafetensorsFile file(fixture_path.string());

    expect(file.contains("encoder.weight"), "weight tensor should be present");
    expect(file.contains("encoder.bias"), "bias tensor should be present");
    expect(!file.contains("missing"), "unknown tensor should not be present");

    const spire::TensorView& weight = file.tensor("encoder.weight");
    expect(weight.shape.size() == 2, "weight should be 2-D");
    expect(weight.shape[0] == fixture.sae_dim, "weight sae_dim");
    expect(weight.shape[1] == fixture.hidden_dim, "weight hidden_dim");
    expect(weight.dtype == spire::DType::F32, "weight dtype should be F32");
    expect(weight.data.size() == fixture.weights.size() * sizeof(float), "weight byte size");

    std::vector<float> weight_values(fixture.weights.size());
    std::memcpy(weight_values.data(), weight.data.data(), weight.data.size());
    expect(weight_values == fixture.weights, "weight bytes should round-trip");

    const spire::TensorView& bias = file.tensor("encoder.bias");
    expect(bias.shape.size() == 1 && bias.shape[0] == fixture.sae_dim, "bias shape");
    expect(bias.dtype == spire::DType::F32, "bias dtype should be F32");

    bool missing_threw = false;
    try {
        (void)file.tensor("missing");
    } catch (const std::exception&) {
        missing_threw = true;
    }
    expect(missing_threw, "missing tensor should throw");

    write_truncated(truncated_path.string());
    bool truncated_threw = false;
    try {
        const spire::SafetensorsFile truncated(truncated_path.string());
        (void)truncated;
    } catch (const std::exception&) {
        truncated_threw = true;
    }
    expect(truncated_threw, "truncated file should throw");

    bool missing_file_threw = false;
    try {
        const spire::SafetensorsFile missing_file(
            (fs::temp_directory_path() / "spire_parser_does_not_exist.safetensors").string());
        (void)missing_file;
    } catch (const std::exception&) {
        missing_file_threw = true;
    }
    expect(missing_file_threw, "nonexistent file should throw");

    fs::remove(fixture_path);
    fs::remove(truncated_path);

    if (failures == 0) {
        std::printf("safetensors parser tests passed\n");
        return 0;
    }
    return 1;
}
