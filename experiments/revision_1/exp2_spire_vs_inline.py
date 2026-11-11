#!/usr/bin/env python3
"""Experiment 2: Spire (async) vs inline SAELens hook on a real model.

Compares forward-pass throughput when SAE encoding is done inline (synchronously
in the hook) vs pushed to Spire (asynchronously). Tests across several
dictionary sizes using synthetic SAEs on GPT-2 small.
"""
import time
import csv

import torch
from transformer_lens import TransformerBridge

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "build/python")
import spire_cpp
from spire.integrations import make_daemon_from_sae, intercept_activations


def make_synthetic_sae(d_in, d_sae, device="cuda"):
    class SAE:
        pass
    sae = SAE()
    sae.W_enc = torch.randn(d_sae, d_in, device=device, dtype=torch.bfloat16)
    sae.b_enc = torch.randn(d_sae, device=device, dtype=torch.float32)
    return sae


def run_inline(model, sae, tokens, n_iters):
    """Forward pass with an inline (synchronous) SAE encode hook."""
    handle = None

    def hook(module, inputs, output):
        act = output.reshape(-1, output.shape[-1])
        latents = torch.relu(act.to(torch.bfloat16) @ sae.W_enc.T + sae.b_enc)
        return output

    handle = model.blocks[0].register_forward_hook(hook)
    # warmup
    with torch.no_grad():
        _ = model(tokens)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    handle.remove()
    return elapsed


def run_spire(model, sae, tokens, n_iters):
    """Forward pass with Spire async hook."""
    daemon = make_daemon_from_sae(sae)

    # warmup
    with intercept_activations(daemon, [model.blocks[0]]):
        with torch.no_grad():
            _ = model(tokens)
    while daemon.try_pop_latents() is not None:
        pass
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        with intercept_activations(daemon, [model.blocks[0]]):
            with torch.no_grad():
                _ = model(tokens)
        # poll to avoid unbounded queue growth
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if daemon.try_pop_latents() is not None:
                break
            time.sleep(0.0001)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


def run_baseline(model, tokens, n_iters):
    """Forward pass with no SAE."""
    with torch.no_grad():
        _ = model(tokens)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(tokens)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    device = "cuda"
    model = TransformerBridge.boot_transformers("gpt2", device=device)
    hidden_dim = model.cfg.d_model
    tokens = torch.randint(0, 50257, (8, 64), device=device)

    d_sae_sizes = [4096, 8192, 16384, 24576]
    n_iters = 20

    results = []
    baseline_elapsed = run_baseline(model, tokens, n_iters)
    baseline_tps = (8 * 64 * n_iters) / baseline_elapsed
    print(f"Baseline: {baseline_elapsed:.4f}s, {baseline_tps:.1f} tok/s")

    for d_sae in d_sae_sizes:
        sae = make_synthetic_sae(hidden_dim, d_sae, device)
        inline_elapsed = run_inline(model, sae, tokens, n_iters)
        spire_elapsed = run_spire(model, sae, tokens, n_iters)
        inline_tps = (8 * 64 * n_iters) / inline_elapsed
        spire_tps = (8 * 64 * n_iters) / spire_elapsed
        results.append({
            "d_sae": d_sae,
            "baseline_tps": baseline_tps,
            "inline_tps": inline_tps,
            "spire_tps": spire_tps,
            "inline_ms_per_iter": inline_elapsed / n_iters * 1000,
            "spire_ms_per_iter": spire_elapsed / n_iters * 1000,
            "speedup_vs_inline": inline_elapsed / spire_elapsed,
        })
        print(f"d_sae={d_sae}: inline={inline_tps:.1f} tok/s, spire={spire_tps:.1f} tok/s, speedup={inline_elapsed/spire_elapsed:.2f}x")

    # Write CSV
    out = "/home/manny/experiments/exp2_spire_vs_inline.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()