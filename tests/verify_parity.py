import json
import os
import struct
import tempfile

import torch
import spire_cpp  # Your compiled C++/CUDA extension


def write_safetensors(path: str, w_enc: torch.Tensor, b_enc: torch.Tensor) -> None:
    """Write the reference weights so the daemon loads exactly what we compare against."""
    weight_bytes = w_enc.cpu().view(torch.int16).numpy().tobytes()
    bias_bytes = b_enc.cpu().numpy().tobytes()
    header = {
        "encoder.weight": {
            "dtype": "BF16",
            "shape": list(w_enc.shape),
            "data_offsets": [0, len(weight_bytes)],
        },
        "encoder.bias": {
            "dtype": "F32",
            "shape": list(b_enc.shape),
            "data_offsets": [len(weight_bytes), len(weight_bytes) + len(bias_bytes)],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        f.write(weight_bytes)
        f.write(bias_bytes)

def pytorch_reference_sae(x: torch.Tensor, w_enc: torch.Tensor, b_enc: torch.Tensor) -> torch.Tensor:
    """
    Computes the reference SAE encoder forward pass in native PyTorch:
    z = ReLU(x * W_enc^T + b_enc)
    """
    # Ensure inputs are float32 for precise mathematical comparison, 
    # or keep bfloat16 to test hardware precision limits.
    linear = torch.matmul(x, w_enc.t()) + b_enc
    return torch.relu(linear)

def run_parity_test():
    print("[*] Setting up numerical parity test harness...")
    
    # Dimensions (matching your typical SAE config)
    batch_size = 4
    hidden_dim = 4096
    sae_dim = 16384
    
    device = torch.device("cuda")
    
    # 1. Generate synthetic test weights and input in bfloat16
    # (In production, these come from your loaded safetensors file)
    w_enc_ref = torch.randn(sae_dim, hidden_dim, device=device, dtype=torch.bfloat16)
    b_enc_ref = torch.randn(sae_dim, device=device, dtype=torch.float32) # Biases are typically float32
    x_input = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.bfloat16)
    
    print(f"[+] Input tensor shape: {x_input.shape} ({x_input.dtype})")
    print(f"[+] Weights shape: {w_enc_ref.shape} ({w_enc_ref.dtype})")

    # 2. Compute Reference PyTorch Output
    print("[*] Computing reference output via PyTorch...")
    z_pytorch = pytorch_reference_sae(x_input, w_enc_ref, b_enc_ref)

    # 3. Initialize Spire Daemon
    # Write the same reference weights to a checkpoint so the daemon loads
    # exactly the tensors used by the PyTorch reference above.
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        checkpoint_path = tmp.name
    write_safetensors(checkpoint_path, w_enc_ref, b_enc_ref)
    print(f"[+] Wrote parity checkpoint: {checkpoint_path}")

    daemon = spire_cpp.SpireDaemon(checkpoint_path)

    print("[*] Executing forward pass through Spire C++/CUDA daemon...")
    # Synchronous testing helper: runs the kernel and returns the latents.
    z_spire = daemon.process_sync(x_input)

    print("[*] Comparing floating-point outputs...")
    
    # 4. Numerical Comparison
    # bfloat16 has lower mantissa precision than float32, so tolerances must account for hardware rounding.
    is_close = torch.allclose(z_spire.float(), z_pytorch.float(), atol=1e-2, rtol=1e-2)
    
    max_absolute_diff = torch.max(torch.abs(z_spire.float() - z_pytorch.float())).item()
    mean_absolute_diff = torch.mean(torch.abs(z_spire.float() - z_pytorch.float())).item()

    print("\n================ PARITY TEST RESULTS ================")
    print(f"Match within tolerance (atol=1e-2, rtol=1e-2): {is_close}")
    print(f"Max Absolute Difference:  {max_absolute_diff:.6f}")
    print(f"Mean Absolute Difference: {mean_absolute_diff:.6f}")
    print("=====================================================")

    if is_close:
        print("[SUCCESS] Spire CUDA kernel matches PyTorch reference math perfectly!")
    else:
        print("[WARNING] Numerical divergence detected. Check register casting or fused ReLU logic.")

    os.unlink(checkpoint_path)

if __name__ == "__main__":
    run_parity_test()