add_executable(safetensors_parser_test safetensors_parser_test.cpp)
target_link_libraries(safetensors_parser_test
    PRIVATE spire_core CUDA::cudart nlohmann_json::nlohmann_json)
add_test(NAME safetensors_parser COMMAND safetensors_parser_test)

add_executable(safetensors_loader_test safetensors_loader_test.cu)
target_link_libraries(safetensors_loader_test
    PRIVATE spire_core CUDA::cudart nlohmann_json::nlohmann_json)
add_test(NAME safetensors_loader COMMAND safetensors_loader_test)

# End-to-end decoder check: verifies x' = z * W_dec + b_dec against a host
# reference for both the vectorized and scalar-fallback kernel paths.
add_executable(sae_decoder_test sae_decoder_test.cu)
target_link_libraries(sae_decoder_test PRIVATE spire_core CUDA::cudart)
add_test(NAME sae_decoder COMMAND sae_decoder_test)

# End-to-end kernel check: the harness verifies SAE latents against a host
# reference and returns non-zero on mismatch.
add_test(NAME sae_encoder COMMAND spire)
