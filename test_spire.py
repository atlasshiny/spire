import torch
import spire_cpp

print("[*] Initializing Spire Daemon...")
# Pass your safetensors weights path here
daemon = spire_cpp.SpireDaemon("spire_sae_weights.safetensors")

print("[*] Creating mock LLM hidden state tensor on GPU...")
# Simulate batch_size=2, hidden_dim=4096 in bfloat16
hidden_states = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)

print("[*] Pushing activation into Spire's lock-free SPSC ring buffer...")
daemon.push_activation(hidden_states)

print("[+] Success! Activation handed off asynchronously to the C++/CUDA daemon.")