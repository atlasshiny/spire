#!/bin/bash
# Nsight Compute profiling for Spire SAE kernels.
#
# Profiles individual CUDA kernels at the instruction level: occupancy,
# memory throughput, compute throughput, register usage, and more.
# Writes a .ncu-rep report openable in the Nsight Compute GUI.
#
# Usage:
#   profiles/ncu_profile.sh [kernel_filter] [output_name]
#
#   kernel_filter : regex to match kernel names (default: bias_|f32_to|kernel).
#                   Known kernel names: kernel (legacy encoder), bias_relu_f32_kernel,
#                   bias_add_bf16_kernel, f32_to_bf16_kernel, plus cuBLAS internals.
#                   Pass "" to profile all kernels. Prefix with "regex:" is added
#                   automatically.
#   output_name   : base name for the report file (default: ncu_profile)
#
# Examples:
#   profiles/ncu_profile.sh "bias_relu"             # profile bias+ReLU epilogue
#   profiles/ncu_profile.sh "f32_to_bf16"           # profile FP32->BF16 conversion
#   profiles/ncu_profile.sh "" full_profile         # profile everything
#
# Output: profiles/<output_name>.ncu-rep
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

BUILD_DIR="${BUILD_DIR:-build/local}"
BENCH_DIR="$BUILD_DIR/benchmarks"
OUTPUT_DIR="${OUTPUT_DIR:-profiles}"
mkdir -p "$OUTPUT_DIR"

KERNEL_FILTER="${1:-bias_|f32_to|kernel}"
OUTPUT_NAME="${2:-ncu_profile}"

# ncu requires a target executable. Use the SAE GEMM benchmark as the default
# target since it exercises both encoder and decoder kernels across batch sizes.
EXE="$BENCH_DIR/sae_gemm_benchmark"
if [ ! -x "$EXE" ]; then
    echo "Benchmark executable not found: $EXE"
    echo "Run 'make build-local' first."
    exit 1
fi

REPORT="$OUTPUT_DIR/$OUTPUT_NAME"

echo "=== Nsight Compute Profiling ==="
echo "Target:        $EXE"
echo "Kernel filter: ${KERNEL_FILTER:-<all>}"
echo "Output:        $REPORT.ncu-rep"
echo ""

# ncu options:
#   --kernel-name-base=function  : match on function name (not mangled)
#   --kernel-name=$KERNEL_FILTER : regex filter for kernel names
#   --launch-skip=2              : skip warm-up launches
#   --launch-count=1             : profile one launch per kernel variant
#   --set=full                   : collect all metrics (comprehensive but slow)
#   --force-overwrite            : overwrite existing report
#   --output=$REPORT             : output path (without extension)
NCU_ARGS=(
    --kernel-name-base=function
    --launch-skip=2
    --launch-count=1
    --set=full
    --force-overwrite
    --export="$REPORT"
)

if [ -n "$KERNEL_FILTER" ]; then
    NCU_ARGS+=(--kernel-name="regex:$KERNEL_FILTER")
fi

ncu "${NCU_ARGS[@]}" "$EXE" 2>&1 | tail -30

echo ""
echo "Report written to: $REPORT.ncu-rep"
echo "Open with: ncu-ui $REPORT.ncu-rep"