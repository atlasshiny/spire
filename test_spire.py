import time

import torch
import spire_cpp

print("[*] Initializing Spire Daemon...")
# Pass your safetensors weights path here
daemon = spire_cpp.SpireDaemon("spire_sae_weights.safetensors")

print("[*] Creating mock LLM hidden state tensor on GPU...")
# Simulate batch_size=2, hidden_dim=4096 in bfloat16
hidden_states = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)

print("[*] Running the synchronous reference path...")
sync_latents = daemon.process_sync(hidden_states)

print("[*] Pushing activation into Spire's lock-free SPSC ring buffer...")
daemon.push_activation(hidden_states)

print("[*] Polling for asynchronous latents...")
async_latents = None
for _ in range(500):
    async_latents = daemon.try_pop_latents()
    if async_latents is not None:
        break
    time.sleep(0.01)

if async_latents is None:
    raise RuntimeError("Timed out waiting for asynchronous latents")

torch.testing.assert_close(async_latents, sync_latents, atol=1e-5, rtol=1e-5)

print(
    f"[+] Success! Async latents {tuple(async_latents.shape)} match the "
    "synchronous reference."
)
