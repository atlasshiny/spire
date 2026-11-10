import time
import torch
import numpy as np
from sae_lens import SAE
from spire import SpireDaemon
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

    # 3. Run forward pass with interception
    tokens = model.to_tokens("The capital of France is Paris")
    
    print("[*] Running forward pass with interception...")
    with intercept_activations(daemon, [model.blocks[0]]):
        _ = model(tokens)

    # 4. Verify results
    latents = daemon.try_pop_latents()
    
    if latents is None:
        raise RuntimeError("Failed to pop latents from daemon!")
    
    print(f"[+] Successfully popped latents. Shape: {latents.shape}")
    assert latents.device.type == "cuda"
    assert latents.dtype == torch.float32
    
    print("\n[SUCCESS] Integration test passed!")

if __name__ == "__main__":
    test_integration()
