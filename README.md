# Spire

Real-time SAE activation interception for GPU inference.

## Building the C++ library

```bash
cmake --preset local
cmake --build --preset local
ctest --test-dir build/local --output-on-failure
```

## Python bindings

The bindings are a compiled extension (`spire_cpp`) built with scikit-build-core.
They need an NVIDIA GPU, the CUDA toolkit, and PyTorch for the target Python
environment.

```bash
uv sync
uv run python -c "import spire_cpp; print(spire_cpp.__doc__)"
```

`uv sync` resolves the lockfile, installs PyTorch and the dev build tools, then
builds `spire_cpp` without build isolation (see
`no-build-isolation-package` in `pyproject.toml`). Building without isolation is
required so CMake can query the environment's own PyTorch for its include and
library paths.

Editable installs are configured with `editable.rebuild = true`, so changes to
the C++ sources are rebuilt incrementally the next time the module is imported.
No reinstall step is required.

If you prefer the low-level pip interface:

```bash
uv pip install torch scikit-build-core
uv pip install -e . --no-build-isolation
```

### Building a wheel

```bash
uv build --wheel --no-build-isolation
```

The resulting wheel contains `spire_cpp.so` at the root of `site-packages`.

### Tests

```bash
# C++ suite
ctest --test-dir build/local --output-on-failure

# End-to-end CUDA vs. PyTorch numerical parity (prints the checkpoint it uses)
uv run python tests/verify_parity.py
```
