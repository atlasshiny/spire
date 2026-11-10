import time

import torch

from spire.integrations import make_daemon_from_sae, intercept_activations


def test_integration():
    print("[*] Starting Spire Integration Test...")

    # 1. Setup model and a small SAE for testing
    device = "cuda"
    from transformer_lens import TransformerBridge

    model = TransformerBridge.boot_transformers("gpt2-small", device=device)
    hidden_dim = model.cfg.d_model  # Usually 768 for gpt2-small

    # Create a synthetic SAE that perfectly matches the model dimensions
    # to avoid dimension mismatch errors during the integration test.
    sae_dim = 2048

    class SyntheticSAE:
        def __init__(self, d_in, d_sae):
            self.W_enc = torch.randn(d_sae, d_in, device=device, dtype=torch.bfloat16)
            self.b_enc = torch.randn(d_sae, device=device, dtype=torch.float32)

    sae = SyntheticSAE(hidden_dim, sae_dim)
    print(f"[+] Created synthetic SAE with d_in={hidden_dim}, d_sae={sae_dim}")

    # 2. Initialize Spire Daemon
    daemon = make_daemon_from_sae(sae)
    print("[+] Spire Daemon initialized from SAE object.")

    # 3. Run forward pass with interception. A plain hook records the same
    # activation so the latents can be checked numerically afterwards.
    captured = {}

    def capture(module, inputs, output):
        captured["activation"] = output.detach()
        return output

    handle = model.blocks[0].register_forward_hook(capture)
    try:
        tokens = model.to_tokens("The capital of France is Paris")

        print("[*] Running forward pass with interception...")
        with intercept_activations(daemon, [model.blocks[0]]):
            _ = model(tokens)
    finally:
        handle.remove()

    # 4. The result is published asynchronously; poll until it appears.
    latents = None
    deadline = time.monotonic() + 5.0
    while latents is None and time.monotonic() < deadline:
        latents = daemon.try_pop_latents()
        if latents is None:
            time.sleep(0.001)

    if latents is None:
        raise RuntimeError("Failed to pop latents from daemon!")

    print(f"[+] Successfully popped latents. Shape: {latents.shape}")
    assert latents.device.type == "cuda"
    assert latents.dtype == torch.float32

    # 5. Compare against the reference encoder on the same activation.
    activation = captured["activation"].reshape(-1, hidden_dim)
    reference = torch.relu(
        activation.to(torch.bfloat16) @ sae.W_enc.T + sae.b_enc
    )
    torch.testing.assert_close(latents, reference, atol=2e-2, rtol=2e-2)
    print(
        f"[+] Latents match the reference (max abs diff "
        f"{(latents - reference).abs().max().item():.6f})"
    )

    print("\n[SUCCESS] Integration test passed!")


if __name__ == "__main__":
    test_integration()
