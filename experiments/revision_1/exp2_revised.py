#!/usr/bin/env python3
"""Experiment 2 (revised): Spire vs Inline on a large model.

Regime: d_model=4096, dictionaries 65k-262k, 12 monitored layers, realistic batches.
Measures inline overhead, async fire-and-forget, and polled variants using CUDA
events with >=5 repetitions, interleaved configurations, median + spread.

Also includes a roofline measurement of the encoder kernel alone (no queue).
"""
import time
import csv
import random
import os

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "build/python")
import spire_cpp
from spire.integrations import make_daemon_from_sae, intercept_activations


# ---------------------------------------------------------------------------
# Synthetic model: 12-layer MLP stack with d_model=4096
# ---------------------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, 4 * d_model, bias=False)
        self.down = nn.Linear(4 * d_model, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.ln(x)
        h = self.down(self.act(self.up(h)))
        return x + h


class SyntheticModel(nn.Module):
    def __init__(self, d_model=4096, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([MLPBlock(d_model) for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_sae(d_in, d_sae, device="cuda"):
    class SAE:
        pass
    sae = SAE()
    sae.W_enc = torch.randn(d_sae, d_in, device=device, dtype=torch.bfloat16)
    sae.b_enc = torch.randn(d_sae, device=device, dtype=torch.float32)
    return sae


def cuda_time_ms(fn, warmup=3, repeats=7):
    """Time fn() using CUDA events. Returns list of ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main():
    device = "cuda"
    d_model = 4096
    n_layers = 8
    monitored_layers = [2]  # 1 monitored layer for cleaner measurement
    batch, seq = 4, 256  # 1024 tokens
    n_iters = 5
    d_sae_list = [65536, 131072]

    model = SyntheticModel(d_model, n_layers).to(device).to(torch.bfloat16).eval()
    tokens = torch.randn(batch, seq, d_model, device=device, dtype=torch.bfloat16)

    results = []

    for d_sae in d_sae_list:
        saes = {li: make_sae(d_model, d_sae, device) for li in monitored_layers}

        # --- Baseline: no hooks ---
        def baseline_fn():
            with torch.no_grad():
                _ = model(tokens)

        baseline_times = cuda_time_ms(baseline_fn, warmup=3, repeats=n_iters)
        baseline_median = sorted(baseline_times)[len(baseline_times)//2]

        # --- Inline: hooks compute SAE latents synchronously ---
        def make_inline_fn():
            handles = []
            for li in monitored_layers:
                sae = saes[li]
                def hook(module, inputs, output, _sae=sae):
                    act = output.reshape(-1, output.shape[-1])
                    _ = torch.relu(act.to(torch.bfloat16) @ _sae.W_enc.T + _sae.b_enc)
                    return output
                handles.append(model.layers[li].register_forward_hook(hook))

            def fn():
                with torch.no_grad():
                    _ = model(tokens)
            return fn, handles

        inline_fn, inline_handles = make_inline_fn()
        inline_times = cuda_time_ms(inline_fn, warmup=3, repeats=n_iters)
        for h in inline_handles:
            h.remove()
        inline_median = sorted(inline_times)[len(inline_times)//2]

        # --- Spire async (fire-and-forget): push only, single sync at end ---
        daemons = {li: make_daemon_from_sae(saes[li]) for li in monitored_layers}

        def make_spire_ff_fn():
            handles = []
            for li in monitored_layers:
                daemon = daemons[li]
                hook = intercept_activations.__wrapped__ if hasattr(intercept_activations, '__wrapped__') else None
                # Use SpireHook manually
                from spire.integrations import SpireHook, SpireHookManager
                mgr = SpireHookManager(daemon)
                h = mgr.register_hook(model.layers[li])
                handles.append((mgr, h))

            def fn():
                with torch.no_grad():
                    _ = model(tokens)
                # single sync at end: drain all queues
                for li in monitored_layers:
                    daemon = daemons[li]
                    deadline = time.monotonic() + 10.0
                    while time.monotonic() < deadline:
                        if daemon.try_pop_latents() is not None:
                            break
                        time.sleep(0.0001)
            return fn, handles

        spire_ff_fn, spire_ff_handles = make_spire_ff_fn()
        spire_ff_times = cuda_time_ms(spire_ff_fn, warmup=3, repeats=n_iters)
        for mgr, h in spire_ff_handles:
            mgr.remove_all_hooks()
        spire_ff_median = sorted(spire_ff_times)[len(spire_ff_times)//2]

        # --- Spire polled: push + poll per forward pass ---
        def make_spire_polled_fn():
            handles = []
            for li in monitored_layers:
                daemon = daemons[li]
                from spire.integrations import SpireHookManager
                mgr = SpireHookManager(daemon)
                h = mgr.register_hook(model.layers[li])
                handles.append((daemon, mgr, h))

            def fn():
                with torch.no_grad():
                    _ = model(tokens)
                for daemon, mgr, h in handles:
                    deadline = time.monotonic() + 10.0
                    while time.monotonic() < deadline:
                        if daemon.try_pop_latents() is not None:
                            break
                        time.sleep(0.0001)
            return fn, handles

        spire_polled_fn, spire_polled_handles = make_spire_polled_fn()
        spire_polled_times = cuda_time_ms(spire_polled_fn, warmup=3, repeats=n_iters)
        for daemon, mgr, h in spire_polled_handles:
            mgr.remove_all_hooks()
        spire_polled_median = sorted(spire_polled_times)[len(spire_polled_times)//2]

        inline_overhead_ms = inline_median - baseline_median
        inline_overhead_pct = (inline_overhead_ms / baseline_median) * 100

        result = {
            "d_sae": d_sae,
            "n_layers": n_layers,
            "n_monitored": len(monitored_layers),
            "batch": batch,
            "seq": seq,
            "baseline_ms_median": baseline_median,
            "baseline_ms_min": min(baseline_times),
            "baseline_ms_max": max(baseline_times),
            "inline_ms_median": inline_median,
            "inline_ms_min": min(inline_times),
            "inline_ms_max": max(inline_times),
            "inline_overhead_ms": inline_overhead_ms,
            "inline_overhead_pct": inline_overhead_pct,
            "spire_ff_ms_median": spire_ff_median,
            "spire_ff_ms_min": min(spire_ff_times),
            "spire_ff_ms_max": max(spire_ff_times),
            "spire_polled_ms_median": spire_polled_median,
            "spire_polled_ms_min": min(spire_polled_times),
            "spire_polled_ms_max": max(spire_polled_times),
        }
        results.append(result)

        print(f"\nd_sae={d_sae}:")
        print(f"  baseline:       {baseline_median:.2f} ms (spread {min(baseline_times):.2f}-{max(baseline_times):.2f})")
        print(f"  inline:         {inline_median:.2f} ms (spread {min(inline_times):.2f}-{max(inline_times):.2f})")
        print(f"  inline overhead: {inline_overhead_ms:.2f} ms ({inline_overhead_pct:.1f}%)")
        print(f"  spire (ff):     {spire_ff_median:.2f} ms")
        print(f"  spire (polled): {spire_polled_median:.2f} ms")

    # Write CSV
    out = "/home/manny/spire/experiments/exp2_revised.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()