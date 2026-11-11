# Spire Build and Test Wrapper

# Configuration
BUILD_DIR_LOCAL = build/local
BUILD_DIR_PYTHON = build/python
PYTHON_BIN = .venv/bin/python
PYTHON_PATH = src

.PHONY: all clean build-local build-python test-local test-integration benchmark bench_and_plot

# Default target: rebuild everything and run all tests
all: build-local build-python test-local test-integration

# Build the C++ core and CLI tools
build-local:
	cmake --build $(BUILD_DIR_LOCAL) --preset local

# Build the Python bindings module
build-python:
	cmake --build $(BUILD_DIR_PYTHON) --target spire_cpp

# Run C++ unit tests
test-local: build-local
	ctest --test-dir $(BUILD_DIR_LOCAL) --output-on-failure

# Run Python integration tests
test-integration: build-python
	PYTHONPATH=$(PYTHON_PATH) $(PYTHON_BIN) tests/integration_test.py

# Run performance benchmarks
benchmark: build-local
	./$(BUILD_DIR_LOCAL)/benchmarks/sae_gemm_benchmark --benchmark_min_time=0.05s

# Run full benchmark + plot pipeline across every benchmark executable
bench_and_plot: build-local
	./benchmarks/bench_and_plot.sh

# Run Nsight Systems profiling (full timeline)
# Usage: make nsys_profile [PROFILE_TARGET=sae_gemm] [PROFILE_OUTPUT=name]
nsys_profile:
	./profiles/nsys_profile.sh $(PROFILE_TARGET) $(PROFILE_OUTPUT)

# Run Nsight Compute profiling (per-kernel metrics)
# Usage: make ncu_profile [PROFILE_KERNEL=filter] [PROFILE_OUTPUT=name]
ncu_profile:
	./profiles/ncu_profile.sh $(PROFILE_KERNEL) $(PROFILE_OUTPUT)

clean:
	rm -rf $(BUILD_DIR_LOCAL) $(BUILD_DIR_PYTHON) benchmark_results plots profiles/*.nsys-rep profiles/*.ncu-rep
