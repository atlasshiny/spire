#!/usr/bin/env python3
"""Experiment 1: Profile whether Spire's async path overlaps with inference.

Runs GPT-2 small with a Spire hook on block 0, performing multiple forward
passes while polling for latents. Designed to be run under Nsight Systems to
visualize CPU/GPU overlap.
"""
import time

import torch
from transformer_lens import TransformerBridge

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "build/python")
import spire_cpp
from spire.integrations import make_daemon_from_sae, intercept_activations


def main():
    device = "cuda"
    model = TransformerBridge.boot_transformers("gpt2-small", device=device)
    hidden_dim = model.cfg.d_model

    class SyntheticSAE:
        def __init__(self, d_in, d_sae):
            self.W_enc = torch.randn(d_sae, d_in, device=device, dtype=torch.bfloat16)
            self.b_enc = torch.randn(d_sae, device=device, dtype=torch.float32)

    sae = SyntheticSAE(hidden_dim, 16384)
    daemon = make_daemon_from_sae(sae)

    tokens = torch.randint(0, 50257, (8, 64), device=device)

    # Warmup
    with intercept_activations(daemon, [model.blocks[0]]):
        _ = model(tokens)
    # drain
    while daemon.try_pop_latents() is not None:
        pass

    # Benchmark iterations
    for i in range(20):
        with intercept_activations(daemon, [model.blocks[0]]):
            _ = model(tokens)
        # poll
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if daemon.try_pop_latents() is not None:
                break
            time.sleep(0.0001)

    torch.cuda.synchronize()
    print("Experiment 1 complete.")


if __name__ == "__main__":
    main()