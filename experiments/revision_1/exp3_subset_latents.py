#!/usr/bin/env python3
"""Experiment 3: Cost savings from computing only a subset of latents.

Measures encoder latency for full vs. partial dictionary sizes. Since the
encoder cost scales linearly with sae_dim, computing only a subset (e.g., the
first K latents of a 24576-dim SAE) saves a proportional fraction of compute.
"""
import time
import csv

import torch

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "build/python")
import spire_cpp


def bench_encoder(hidden_dim, sae_dim, batch_tokens, n_iters=50):
    weights = torch.randn(sae_dim, hidden_dim, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(sae_dim, device="cuda", dtype=torch.float32)
    daemon = spire_cpp.SpireDaemon(weights, bias)

    inp = torch.randn(batch_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    # warmup
    daemon.push_activation(inp)
    while daemon.try_pop_latents() is None:
        pass

    start = time.perf_counter()
    for _ in range(n_iters):
        daemon.push_activation(inp)
        while daemon.try_pop_latents() is None:
            pass
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_iters * 1000  # ms per call


def main():
    hidden_dim = 768
    batch_tokens = 512
    full_dim = 24576
    subsets = [full_dim, full_dim // 2, full_dim // 4, full_dim // 8, full_dim // 16, 4096]

    results = []
    for d in subsets:
        ms = bench_encoder(hidden_dim, d, batch_tokens)
        results.append({"sae_dim": d, "ms_per_call": ms, "fraction_of_full": d / full_dim})
        print(f"sae_dim={d:>6}  ms/call={ms:.4f}  fraction={d/full_dim:.4f}")

    # Compute savings relative to full
    full_ms = results[0]["ms_per_call"]
    for r in results:
        r["savings_pct"] = (1 - r["ms_per_call"] / full_ms) * 100

    out = "/home/manny/experiments/exp3_subset_latents.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()