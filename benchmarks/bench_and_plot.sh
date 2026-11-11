#!/bin/bash
# Run every Spire benchmark, capture CSV output, and generate plots.
#
# Usage:
#   benchmarks/bench_and_plot.sh [output_root]
#
# Each benchmark executable is run independently so one failure (for example an
# OOM on a very large shape) does not abort the rest. CSVs land in
# ./benchmark_results and figures in ./plots/<benchmark>/.
set -u

# Resolve paths relative to the repository root so the script works no matter
# where it is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

BUILD_DIR="${BUILD_DIR:-build/local}"
BENCH_DIR="$BUILD_DIR/benchmarks"
RESULTS_DIR="${RESULTS_DIR:-benchmark_results}"
PLOT_DIR="${1:-plots}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

echo "--- Starting Spire benchmark + plot pipeline ---"
echo "Build dir:     $BUILD_DIR"
echo "Results dir:   $RESULTS_DIR"
echo "Plot dir:      $PLOT_DIR"
echo "Python:        $PYTHON_BIN"

if [ ! -d "$BENCH_DIR" ]; then
    echo "Benchmark directory not found: $BENCH_DIR"
    echo "Run 'make build-local' first."
    exit 1
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "$PLOT_DIR"

BENCHMARKS=(
    sae_gemm_benchmark
    sae_kernel_benchmark
    sae_decoder_benchmark
    pipeline_benchmark
    activation_queue_benchmark
    allocation_benchmark
)

failed=()
for bench in "${BENCHMARKS[@]}"; do
    exe="$BENCH_DIR/$bench"
    csv="$RESULTS_DIR/$bench.csv"

    if [ ! -x "$exe" ]; then
        echo "Skipping $bench (not built at $exe)"
        failed+=("$bench (missing)")
        continue
    fi

    echo ""
    echo ">>> Running $bench ..."
    if "$exe" --benchmark_format=csv --benchmark_min_time=0.1s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true > "$csv" 2> "$RESULTS_DIR/$bench.stderr"; then
        rows=$(( $(wc -l < "$csv") - 1 ))
        echo "    wrote $csv ($rows data rows)"
    else
        status=$?
        echo "    FAILED (exit $status); see $RESULTS_DIR/$bench.stderr"
        # Keep any partial CSV so the plotter can still use what succeeded.
        failed+=("$bench (exit $status)")
    fi
done

echo ""
echo ">>> Generating plots ..."
"$PYTHON_BIN" "$SCRIPT_DIR/plot.py" "$RESULTS_DIR" "$PLOT_DIR"
plot_status=$?

echo ""
if [ ${#failed[@]} -eq 0 ]; then
    echo "All benchmarks completed successfully."
else
    echo "The following benchmarks had problems:"
    for item in "${failed[@]}"; do
        echo "  - $item"
    done
fi

echo "Plots are in: $PLOT_DIR"
echo "Raw CSVs are in: $RESULTS_DIR"
exit $plot_status
