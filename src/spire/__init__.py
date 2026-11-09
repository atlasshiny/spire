"""Spire Python integration utilities.

This package re-exports the compiled ``spire_cpp`` extension as ``spire`` and
provides integration helpers for SAELens and TransformerLens in
:mod:`spire.integrations`.

Example::

    import spire
    from spire.integrations import make_daemon_from_sae

    daemon = make_daemon_from_sae(sae_object)
"""

from .integrations import (
    SpireHook,
    SpireHookManager,
    extract_saelens_weights,
    extract_sae_weights,
    extract_transformerlens_weights,
    intercept_activations,
    make_daemon_from_sae,
    make_daemon_from_tensors,
)

# Re-export the compiled extension so that ``import spire`` gives access to
# SpireDaemon and the low-level bindings.
import spire_cpp as _spire_cpp

SpireDaemon = _spire_cpp.SpireDaemon

__all__ = [
    "SpireDaemon",
    "SpireHook",
    "SpireHookManager",
    "extract_saelens_weights",
    "extract_sae_weights",
    "extract_transformerlens_weights",
    "intercept_activations",
    "make_daemon_from_sae",
    "make_daemon_from_tensors",
]
