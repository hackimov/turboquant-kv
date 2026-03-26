"""
Optional integration helpers for Hugging Face ``transformers`` (KV cache) and notes for vLLM.

- **Legacy tuple** ``past_key_values``: ``TurboQuantModel.quantize_past_key_values`` / ``dequantize_past_key_values``.
- **HF ``Cache`` (5.x):** ``turboquant_dynamic_cache(model.config, quantizer)`` or
  ``TurboQuantModel.make_dynamic_cache()`` → ``TurboQuantDynamicCache`` with compressed non-sliding layers;
  for encoder-decoder models: ``turboquant_encoder_decoder_cache(...)`` or
  ``TurboQuantModel.make_encoder_decoder_cache()`` → ``EncoderDecoderCache`` with compressed cross-attention KV;
  paged export: ``export_cache_to_paged_per_layer`` in ``turboquant.hf_cache``.

Requires ``pip install transformers`` (or ``turboquant-kv[hf]``) for the cache path; importing ``turboquant.hf_cache``
pulls in ``transformers``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .core import TurboQuantProd


PastKeyValues = Sequence[Tuple[torch.Tensor, torch.Tensor]]


class TurboQuantModel:
    """
    Wrapper that helps quantize/dequantize the KV cache.

    It does not rewrite attention logic automatically; that requires a concrete
    integration for your `past_key_values` layout in your `transformers`/vLLM version.
    """

    def __init__(
        self,
        model: Any,
        quantizer: Optional[TurboQuantProd] = None,
        *,
        bits: float = 3.0,
        head_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model

        if quantizer is not None:
            self.quantizer = quantizer
            return

        inferred_head_dim = head_dim
        cfg = getattr(model, "config", None)
        if inferred_head_dim is None and cfg is not None:
            # Common HF naming: hidden_size / num_attention_heads
            hidden_size = getattr(cfg, "hidden_size", None)
            num_heads = getattr(cfg, "num_attention_heads", None)
            if hidden_size is not None and num_heads not in (None, 0):
                try:
                    inferred_head_dim = int(hidden_size) // int(num_heads)
                except Exception:
                    inferred_head_dim = None

            # Some architectures expose `head_dim` directly.
            if inferred_head_dim is None:
                inferred_head_dim = getattr(cfg, "head_dim", None)

        if inferred_head_dim is None:
            # Reasonable default for many LLMs; override for your architecture.
            inferred_head_dim = 128

        self.quantizer = TurboQuantProd(
            bits=bits,
            head_dim=int(inferred_head_dim),
            device=device,
            dtype=dtype,
        )

    def quantize_past_key_values(
        self, past_key_values: PastKeyValues, *, return_compressed: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """
        Quantize each layer's (k, v) tensors using `TurboQuantProd.quantize_kv_cache`.

        Expected tensor shapes (typical HF/vLLM):
          - k: [batch, heads, seq_len, head_dim]
          - v: [batch, heads, seq_len, head_dim]
        """
        out: List[Dict[str, torch.Tensor]] = []
        for layer_k, layer_v in past_key_values:
            out.append(
                self.quantizer.quantize_kv_cache(
                    layer_k,
                    layer_v,
                    return_compressed=return_compressed,
                )
            )
        return tuple(out)

    def dequantize_past_key_values(
        self, compressed_past_key_values: Iterable[Dict[str, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Inverse of `quantize_past_key_values` for compressed dictionary format.
        """
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_dict in compressed_past_key_values:
            k, v = self.quantizer.decompress(layer_dict)
            out.append((k, v))
        return tuple(out)

    def make_dynamic_cache(
        self,
        *,
        triton_fused_layers: bool = False,
        strict_reencode: bool = False,
        hybrid_float_cache: bool = False,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        """
        Build a ``TurboQuantDynamicCache`` for ``past_key_values`` (``transformers`` 5.x ``Cache`` API).

        Set ``triton_fused_layers=True`` for :class:`TurboQuantTritonFusedCacheLayer`, then call
        ``enable_llama_fused_attention()`` to replace ``LlamaAttention`` modules with
        :class:`~turboquant.hf_llama_fused.TurboQuantLlamaAttention` (no monkey-patch; Triton on CUDA,
        or SDPA fallback on Apple Metal/MPS for the fused path).

        Set ``hybrid_float_cache=True`` to skip full-sequence decompress on each ``update`` on the
        non-strict path (see :class:`~turboquant.hf_cache.TurboQuantCacheLayer`).

        Example::

            cache = wrapper.make_dynamic_cache()
            out = model(**inputs, past_key_values=cache, use_cache=True)

        """
        from .hf_cache import turboquant_dynamic_cache

        cfg = getattr(self.model, "config", None)
        if cfg is None:
            raise ValueError("TurboQuantModel.model must have a .config (HF PreTrainedConfig)")
        return turboquant_dynamic_cache(
            cfg,
            self.quantizer,
            triton_fused_layers=triton_fused_layers,
            strict_reencode=strict_reencode,
            hybrid_float_cache=hybrid_float_cache,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

    def make_encoder_decoder_cache(
        self,
        *,
        quantize_self: bool = False,
        triton_fused_self: bool = False,
        triton_fused_cross: bool = False,
        strict_reencode: bool = False,
        hybrid_float_cache: bool = False,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        """
        Build a :class:`turboquant.hf_cache.TurboQuantEncoderDecoderCache` for encoder-decoder models
        (T5 / mT5 / M2M-100 & NLLB-family / MarianMT, etc.) where **cross-attention KV** is TurboQuant-compressed.

        By default `quantize_self=False` keeps decoder self-attention KV as HF float (safer quality parity for
        translation/summarization).
        """
        from .hf_cache import turboquant_encoder_decoder_cache

        cfg = getattr(self.model, "config", None)
        if cfg is None:
            raise ValueError("TurboQuantModel.model must have a .config (HF PreTrainedConfig)")

        return turboquant_encoder_decoder_cache(
            cfg,
            self.quantizer,
            quantize_self=quantize_self,
            triton_fused_self=triton_fused_self,
            triton_fused_cross=triton_fused_cross,
            strict_reencode=strict_reencode,
            hybrid_float_cache=hybrid_float_cache,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

    def enable_decoder_fused_attention(
        self,
        architecture: str = "auto",
        *,
        allow_attention_subclass: bool = False,
    ) -> None:
        """
        Swap in TurboQuant attention wrappers (Triton fused where parity allows; see ``hf_fused_attention``).

        Prefer this name over ``enable_llama_fused_attention`` (legacy alias).
        """
        from .hf_fused_attention import install_turboquant_fused_attention

        install_turboquant_fused_attention(
            self.model,
            self.quantizer,
            architecture=architecture,
            allow_attention_subclass=allow_attention_subclass,
        )

    def disable_decoder_fused_attention(self) -> None:
        from .hf_fused_attention import uninstall_turboquant_fused_attention

        uninstall_turboquant_fused_attention(self.model)

    def enable_llama_fused_attention(
        self,
        architecture: str = "auto",
        *,
        allow_attention_subclass: bool = False,
    ) -> None:
        """Alias of :meth:`enable_decoder_fused_attention`."""
        self.enable_decoder_fused_attention(
            architecture=architecture,
            allow_attention_subclass=allow_attention_subclass,
        )

    def disable_llama_fused_attention(self) -> None:
        self.disable_decoder_fused_attention()

