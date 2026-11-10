import time
import torch
import numpy as np
from sae_lens import SAE
from transformer_lens import TransformerBridge
from spire import SpireDaemon
from spire.integrations import make_daemon_from_sae, intercept_activations

def benchmark_integration():
    print("\n" + "="*50)
    print("SpiRE Integration Performance Benchmark")
    print("="*50)
    
    device = "cuda"
    batch_size = 16
    seq_len = 128
    
    # Setup model and SAE
    model = TransformerBridge.boot_transformers("gpt2-small", device=device)
    hidden_dim = model.cfg.d_model
    sae_dim = 16384
    
    class SyntheticSAE:
        def __init__(self, d_in, d_sae):
            self.W_enc = torch.randn(d_sae, d_in, device=device, dtype=torch.bfloat16)
            self.b_enc = torch.randn(d_sae, device=device, dtype=torch.float32)
    
    sae = SyntheticSAE(hidden_dim, sae_dim)
    print(f"[+] Using synthetic SAE: d_in={hidden_dim}, d_sae={sae_dim}")


    daemon = make_daemon_from_sae(sae)
    
    # Create large synthetic input
    tokens = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    # Warmup
    print("[*] Warming up...")
    with intercept_activations(daemon, [model.blocks[0]]):
        _ = model(tokens)
    daemon.try_pop_latents() # Clear queue

    # Actual Benchmark
    num_iters = 10
    print(f"[*] Benchmarking {num_iters} iterations (Batch={batch_size}, Seq={seq_len})...")
    
    start_time = time.perf_counter()
    for i in range(num_iters):
        with intercept_activations(daemon, [model.blocks[0]]):
            _ = model(tokens)
        
        # Poll for results to ensure we aren't just filling the queue
        # In a real app, this would happen in a separate thread.
        while daemon.try_pop_latents() is None:
            pass
            
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iters
    throughput = (batch_size * seq_len * num_iters) / total_time
    
    print("\nResults:")
    print(f"  Total time:    {total_time:.4f} s")
    print(f"  Average/iter: {avg_time:.4f} s")
    print(f"  Throughput:    {throughput:.2f} tokens/sec")
    print("="*50 + "\n")

if __name__ == "__main__":
    benchmark_integration()
