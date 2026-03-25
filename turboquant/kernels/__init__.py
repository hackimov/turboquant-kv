"""
Triton kernels for TurboQuant
"""

__all__ = ["turboquant_attention"]

# Avoid import-time hard dependency on Triton.
# Unit tests and CPU-only usage should still work without `triton`.
try:
    # Explicit import to make sure the module is registered
    from .quantized_attention import turboquant_attention  # type: ignore
except ModuleNotFoundError as e:
    if e.name != "triton":
        raise

    def turboquant_attention(*args, **kwargs):
        raise ModuleNotFoundError(
            "turboquant.kernels.turboquant_attention requires Triton. "
            "On Windows: pip install triton-windows (or pip install turboquant-kv[triton]). "
            "On Linux: pip install triton (or pip install turboquant-kv[triton])."
        )