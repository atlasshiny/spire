#!/usr/bin/env python3
"""Roofline analysis for the Spire SAE encoder kernel.

Measures encoder throughput (TFLOP/s) and memory bandwidth (GB/s) across a
range of (batch, hidden, sae) configurations using the C++ sae_gemm_benchmark.
Produces a roofline plot showing where the kernel is compute-bound vs
memory-bound.
"""
import subprocess
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS = "/home/manny/spire/experiments"
BUILD = "/home/manny/spire/build/local"
BENCH = os.path.join(BUILD, "benchmarks", "sae_gemm_benchmark")


def run_benchmark():
    """Run the C++ benchmark and return parsed rows."""
    result = subprocess.run(
        [BENCH, "--benchmark_format=csv"],
        capture_output=True, text=True, timeout=300,
    )
    lines = result.stdout.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("name,"))
    import io
    reader = csv.DictReader(lines[header_idx:])
    rows = []
    for r in reader:
        name = r["name"]
        if "Encoder" not in name:
            continue
        if "gemm" not in name.lower():
            continue
        # parse batch/hidden/sae from label like "b=1 h=4096 s=16384 gemm"
        m = re.search(r"b=(\d+)\s+h=(\d+)\s+s=(\d+)", r.get("label", ""))
        if not m:
            continue
        b, h, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        tflops = float(r.get("TFLOP/s", 0))
        gbs = float(r.get("weight GB/s", 0))
        ms = float(r.get("ms/launch", 0))
        rows.append({"batch": b, "hidden": h, "sae": s, "TFLOP/s": tflops,
                     "GB/s": gbs, "ms": ms})
    return rows


def main():
    print("Running sae_gemm_benchmark ...")
    rows = run_benchmark()
    print(f"Parsed {len(rows)} encoder (gemm) rows")

    # Save CSV
    out_csv = os.path.join(EXPERIMENTS, "roofline_encoder.csv")
    with open(out_csv, "w", newline="") as f:
        import csv as c
        w = c.DictWriter(f, fieldnames=["batch", "hidden", "sae", "TFLOP/s", "GB/s", "ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")

    # Roofline plot: arithmetic intensity (FLOP/byte) vs throughput
    # Arithmetic intensity = 2 * batch * sae * hidden / (batch * hidden * 2 + sae * hidden * 2)
    # Simplified: FLOPs / (weight bytes read) = 2*batch*sae*hidden / (sae*hidden*2) = batch
    # Actually AI = FLOPs / bytes = 2*batch*sae*hidden / (sae*hidden*2 + batch*hidden*2) ~ batch (for large sae)
    ais = []
    tflops = []
    labels = []
    for r in rows:
        ai = r["batch"]  # approximate arithmetic intensity (FLOP per weight byte)
        ais.append(ai)
        tflops.append(r["TFLOP/s"])
        labels.append(f"b={r['batch']}")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(ais, tflops, c=[r["sae"] for r in rows], cmap="viridis",
                         s=80, edgecolor="white", linewidth=0.5, alpha=0.85)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("SAE dimension")

    # Roofline ceilings
    peak_compute = 20.0  # RTX 5060 ~20 TFLOP/s (BF16)
    peak_bw = 272.0      # RTX 5060 memory bandwidth ~272 GB/s
    ai_range = np.array([0.5, 1, 2, 4, 8, 16, 32, 64])
    roofline = np.minimum(peak_compute, peak_bw * ai_range / 1000.0)
    ax.plot(ai_range, roofline, "k--", linewidth=1.5, label="Roofline ceiling")
    ax.axhline(peak_compute, color="red", linestyle=":", linewidth=1, label=f"Peak compute ({peak_compute} TFLOP/s)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte, ~batch size)")
    ax.set_ylabel("Throughput (TFLOP/s)")
    ax.set_title("SAE Encoder Roofline (cuBLAS GEMM)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(EXPERIMENTS, "roofline_encoder.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    # Print summary
    print("\nRoofline summary:")
    for r in sorted(rows, key=lambda x: x["TFLOP/s"], reverse=True)[:5]:
        print(f"  b={r['batch']:>2} h={r['hidden']:>5} s={r['sae']:>6}  {r['TFLOP/s']:.2f} TFLOP/s  {r['GB/s']:.1f} GB/s  {r['ms']:.3f} ms")


if __name__ == "__main__":
    main()