#!/bin/bash
# Nsight Systems profiling for Spire.
#
# Captures a full-application timeline (kernel launches, memory transfers, CPU
# threads, CUDA API calls) and writes a .nsys-rep report that can be opened
# in the Nsight Systems GUI.
#
# Usage:
#   profiles/nsys_profile.sh [target] [output_name]
#
#   target      : one of "spire", "sae_gemm", "sae_kernel", "sae_decoder",
#                 "pipeline", "allocation", "activation_queue"
#                 (default: sae_gemm)
#   output_name : base name for the report file (default: <target>_profile)
#
# Output: profiles/<output_name>.nsys-rep
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

BUILD_DIR="${BUILD_DIR:-build/local}"
BENCH_DIR="$BUILD_DIR/benchmarks"
OUTPUT_DIR="${OUTPUT_DIR:-profiles}"
mkdir -p "$OUTPUT_DIR"

TARGET="${1:-sae_gemm}"
OUTPUT_NAME="${2:-${TARGET}_profile}"

# Map target name to executable path
case "$TARGET" in
    spire)
        EXE="$BUILD_DIR/spire"
        ;;
    sae_gemm)
        EXE="$BENCH_DIR/sae_gemm_benchmark"
        ;;
    sae_kernel)
        EXE="$BENCH_DIR/sae_kernel_benchmark"
        ;;
    sae_decoder)
        EXE="$BENCH_DIR/sae_decoder_benchmark"
        ;;
    pipeline)
        EXE="$BENCH_DIR/pipeline_benchmark"
        ;;
    allocation)
        EXE="$BENCH_DIR/allocation_benchmark"
        ;;
    activation_queue)
        EXE="$BENCH_DIR/activation_queue_benchmark"
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Valid targets: spire, sae_gemm, sae_kernel, sae_decoder, pipeline, allocation, activation_queue"
        exit 1
        ;;
esac

if [ ! -x "$EXE" ]; then
    echo "Executable not found: $EXE"
    echo "Run 'make build-local' first."
    exit 1
fi

REPORT="$OUTPUT_DIR/$OUTPUT_NAME"

echo "=== Nsight Systems Profiling ==="
echo "Target:  $TARGET ($EXE)"
echo "Output:  $REPORT.nsys-rep"
echo ""

# nsys profile options:
#   --trace=cuda,nvtx,osrt   : trace CUDA, NVTX, and OS runtime
#   --sample=cpu              : sample CPU stacks
#   --cpuctxsw=none           : skip context switches (reduces overhead)
#   --force-overwrite=true     : overwrite existing report
#   --output=$REPORT          : output path (without extension)
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$REPORT" \
    "$EXE" 2>&1 | tail -20

echo ""
echo "Report written to: $REPORT.nsys-rep"
echo "Open with: nsys-ui $REPORT.nsys-rep"