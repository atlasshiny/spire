"""
Integration utilities for connecting Spire with SAELens and TransformerLens.

All performance-critical operations (weight upload, queue pushing, kernel
launching) happen in C++ via the `spire_cpp` bindings. Python is only used for
one-time weight extraction, hook registration, and result polling.

The hot path — `push_activation()` — releases the GIL (via
`py::call_guard<py::gil_scoped_release>()`) so pushing activations during a
model forward pass does not contend with Python's GIL.

Typical usage:

    from spire.integrations import make_daemon_from_sae, SpireHookManager

    sae = SAE.from_pretrained(release="gpt2-small-res-jb",
                              sae_id="blocks.8.hook_resid_pre",
                              device="cuda")
    daemon = make_daemon_from_sae(sae)

    manager = SpireHookManager(daemon)
    handle = manager.register_hook(model.blocks[8], "resid_pre")

    # Run the model; activations are pushed into Spire as they are produced.
    logits = model(tokens)

    latents = daemon.try_pop_latents()  # [batch, pos, d_sae] FP32 CUDA
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

import spire_cpp  # The compiled Spire C++/CUDA extension


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def _to_cuda_tensors(
    weights: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize extracted SAE tensors to the layout Spire expects.

    Spire expects:
      * weights: [sae_dim, hidden_dim] BF16, contiguous, on CUDA
      * bias:    [sae_dim] FP32, contiguous, on CUDA
    """
    weights = weights.detach().cuda().to(torch.bfloat16).contiguous()
    if bias is None:
        bias = torch.zeros(
            weights.shape[0],
            device=weights.device,
            dtype=torch.float32,
        )
    bias = bias.detach().cuda().to(torch.float32).contiguous()
    return weights, bias


def extract_saelens_weights(sae: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract encoder weights and bias from a SAELens ``SAE`` object.

    SAELens SAEs expose:
      * ``sae.W_enc``: [d_sae, d_in] encoder weights
      * ``sae.b_enc``: [d_sae] encoder bias (present on most architectures)

    Args:
        sae: A SAELens ``SAE`` instance.

    Returns:
        Tuple of (weights_tensor, bias_tensor), both on CUDA.
    """
    if not hasattr(sae, "W_enc"):
        raise AttributeError(
            "SAELens SAE has no 'W_enc' attribute; expected an SAE object"
            f" but got {type(sae).__name__}"
        )
    bias = getattr(sae, "b_enc", None)
    return _to_cuda_tensors(sae.W_enc, bias)


def extract_transformerlens_weights(sae: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract encoder weights and bias from a TransformerLens SAE object.

    TransformerLens SAEs expose:
      * ``sae.W_enc``: [d_sae, d_in] encoder weights
      * ``sae.b_enc``: [d_sae] encoder bias

    Args:
        sae: A TransformerLens SAE instance.

    Returns:
        Tuple of (weights_tensor, bias_tensor), both on CUDA.
    """
    weight_attrs = ["W_enc", "weights", "encoder_weight", "W_encoder"]
    bias_attrs = ["b_enc", "bias", "encoder_bias", "b_encoder"]

    weights = None
    for attr in weight_attrs:
        if hasattr(sae, attr):
            weights = getattr(sae, attr)
            break
    if weights is None:
        raise AttributeError(
            "Could not find encoder weights in TransformerLens SAE object. "
            f"Expected one of: {weight_attrs}"
        )

    bias = None
    for attr in bias_attrs:
        if hasattr(sae, attr):
            bias = getattr(sae, attr)
            break

    return _to_cuda_tensors(weights, bias)


def extract_sae_weights(sae_object: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract encoder weights and bias from a SAE library object.

    This is a generic helper that inspects the object for standard attribute
    names used by SAELens, TransformerLens, and similar libraries. Use
    ``extract_saelens_weights`` or ``extract_transformerlens_weights`` when you
    know which library produced the object.

    Args:
        sae_object: An SAE object from SAELens, TransformerLens, etc.

    Returns:
        Tuple of (weights_tensor, bias_tensor), both on CUDA.
    """
    # Try the library-specific extractors first when the object type is known.
    type_name = type(sae_object).__name__
    module_name = type(sae_object).__module__
    if "sae_lens" in module_name:
        return extract_saelens_weights(sae_object)
    if "transformer_lens" in module_name:
        return extract_transformerlens_weights(sae_object)

    # Fall back to attribute-name heuristics.
    weight_candidates = ["W_enc", "weights", "encoder_weight", "W_encoder"]
    bias_candidates = ["b_enc", "bias", "encoder_bias", "b_encoder"]

    weights = None
    for attr in weight_candidates:
        if hasattr(sae_object, attr):
            weights = getattr(sae_object, attr)
            break

    if weights is None:
        raise AttributeError(
            f"Could not find encoder weights in object of type {type(sae_object)}. "
            f"Expected one of: {weight_candidates}"
        )

    bias = None
    for attr in bias_candidates:
        if hasattr(sae_object, attr):
            bias = getattr(sae_object, attr)
            break

    return _to_cuda_tensors(weights, bias)


# ---------------------------------------------------------------------------
# Daemon construction
# ---------------------------------------------------------------------------

def make_daemon_from_tensors(
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> spire_cpp.SpireDaemon:
    """Create a SpireDaemon directly from PyTorch tensors already in VRAM.

    This is the recommended path for SAELens/TransformerLens integration:

        sae = SAE.from_pretrained(...)
        daemon = make_daemon_from_tensors(sae.W_enc, sae.b_enc)

    Args:
        weights: [sae_dim, hidden_dim] BF16 tensor on CUDA.
        bias:    [sae_dim] FP32 tensor on CUDA.

    Returns:
        Configured SpireDaemon ready to accept activations.
    """
    return spire_cpp.SpireDaemon(weights, bias)


def make_daemon_from_sae(sae: Any) -> spire_cpp.SpireDaemon:
    """Create a SpireDaemon from a SAELens or TransformerLens SAE object.

    Args:
        sae: A SAELens ``SAE`` or TransformerLens SAE instance.

    Returns:
        Configured SpireDaemon ready to accept activations.
    """
    weights, bias = extract_sae_weights(sae)
    return make_daemon_from_tensors(weights, bias)


# ---------------------------------------------------------------------------
# Forward hooks
# ---------------------------------------------------------------------------

class SpireHook:
    """Thin wrapper around a PyTorch forward hook that pushes activations into
    the Spire daemon.

    Usage:

        hook = SpireHook(daemon)
        handle = module.register_forward_hook(hook)
    """

    def __init__(self, daemon: spire_cpp.SpireDaemon):
        self.daemon = daemon

    def __call__(self, module, inputs, output):
        # PyTorch invokes this during the forward pass. push_activation releases
        # the GIL, so the enqueue itself does not contend with the GIL.
        if output is not None:
            processed = output
            if processed.dim() == 3:
                b, s, h = processed.shape
                processed = processed.reshape(b * s, h)
            
            if processed.dtype != torch.bfloat16:
                processed = processed.to(torch.bfloat16)
                
            self.daemon.push_activation(processed)
            
        return output


class SpireHookManager:
    """Manages a collection of SpireHooks registered on model modules.

    Use this to intercept activations from multiple hook points at once and
    poll the resulting latents from a single daemon.

    Usage:

        manager = SpireHookManager(daemon)
        handle_1 = manager.register_hook(model.blocks[8], "resid_pre")
        handle_2 = manager.register_hook(model.blocks[9], "resid_pre")

        logits = model(tokens)

        latents = daemon.try_pop_latents()  # most recent completed latents

        manager.remove_all_hooks()
    """

    def __init__(self, daemon: spire_cpp.SpireDaemon):
        self.daemon = daemon
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def register_hook(self, module: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on ``module`` that pushes its output into
        the Spire daemon. Returns a removable handle."""
        hook = SpireHook(self.daemon)
        handle = module.register_forward_hook(hook)
        self._handles.append(handle)
        return handle

    def remove_all_hooks(self) -> None:
        """Remove every hook registered through this manager."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Convenience context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def intercept_activations(
    daemon: spire_cpp.SpireDaemon,
    modules: Sequence[torch.nn.Module],
):
    """Context manager that registers SpireHooks on ``modules`` for the
    duration of the ``with`` block and removes them on exit.

    Usage:

        daemon = make_daemon_from_sae(sae)
        with intercept_activations(daemon, [model.blocks[8]]):
            logits = model(tokens)
        latents = daemon.try_pop_latents()
    """
    manager = SpireHookManager(daemon)
    for module in modules:
        manager.register_hook(module)
    try:
        yield daemon
    finally:
        manager.remove_all_hooks()
