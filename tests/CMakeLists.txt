add_executable(safetensors_parser_test safetensors_parser_test.cpp)
target_link_libraries(safetensors_parser_test
    PRIVATE spire_core CUDA::cudart nlohmann_json::nlohmann_json)
add_test(NAME safetensors_parser COMMAND safetensors_parser_test)

add_executable(safetensors_loader_test safetensors_loader_test.cu)
target_link_libraries(safetensors_loader_test
    PRIVATE spire_core CUDA::cudart nlohmann_json::nlohmann_json)
add_test(NAME safetensors_loader COMMAND safetensors_loader_test)

# End-to-end kernel check: the harness verifies SAE latents against a host
# reference and returns non-zero on mismatch.
add_test(NAME sae_encoder COMMAND spire)
