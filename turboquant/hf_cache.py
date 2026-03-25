"""
Hugging Face ``transformers`` integration: dynamic KV ``Cache`` with TurboQuant-compressed storage.

Non-sliding layers use ``TurboQuantCacheLayer`` (quantize on each ``update``; return full-precision KV
for standard attention). Optional ``hybrid_float_cache=True`` keeps a running float KV (concatenating
incoming states) so ``update`` skips full ``decompress`` on the non-strict path; see class docstring.
Sliding-window layers stay ``DynamicSlidingWindowLayer`` (HF, FP).

**Paged export:** ``export_compressed_to_paged`` / ``export_cache_to_paged_per_layer`` match
``pack_dense_kv_to_paged`` for our Triton paged API (not a vLLM worker without upstream changes).

Tested against ``transformers`` 5.x (``Cache.update(..., cache_kwargs)``, ``get_mask_sizes(cache_position)``).
Install: ``pip install turboquant-kv[hf]`` or ``pip install transformers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from .core import TurboQuantProd, concat_compressed_kv
from .kernels.fused_attention import pack_dense_kv_to_paged

if TYPE_CHECKING:
    from transformers.configuration_utils import PreTrainedConfig

try:
    from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "Module `turboquant.hf_cache` requires `transformers`. "
        "Install: pip install transformers   or   pip install turboquant-kv[hf]"
    ) from _e


def is_hf_cache_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _text_decoder_config(config: "PreTrainedConfig"):
    if hasattr(config, "get_text_config"):
        return config.get_text_config(decoder=True)
    return config


class TurboQuantCacheLayer(CacheLayerMixin):
    """
    One decoder layer: KV stored as TurboQuant compressed dict; ``keys``/``values`` decode on access.

    Parameters
    ----------
    strict_reencode : bool
        If ``True``, each ``update`` re-quantizes the full float KV history (closer to a single batch
        ``quantize_kv``, slower). If ``False`` (default), only the **new** positions are quantized and
        concatenated along seq (``concat_compressed_kv``) — faster; numerically slightly different.
    hybrid_float_cache : bool
        If ``True`` (default ``False``), keep materialized float K/V by appending each step's
        ``key_states``/``value_states``. **Non-strict:** ``update`` skips full-sequence ``decompress``
        (saves time; uses native float history, so logits can differ from decompressing the lossy
        compressed prefix). **Strict:** caches the previous step's decompressed KV so the prefix is not
        decoded twice per step (still one full decompress per step for the return value). Uses extra
        memory (~full float KV in addition to compressed tensors).
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, quantizer: TurboQuantProd, *, strict_reencode: bool = False, hybrid_float_cache: bool = False):
        # Intentionally no ``CacheLayerMixin.__init__()`` — it assigns ``self.keys``/``.values``,
        # which clashes with our read-only properties backed by ``_compressed``.
        self.quantizer = quantizer
        self.strict_reencode = bool(strict_reencode)
        self.hybrid_float_cache = bool(hybrid_float_cache)
        self._compressed: Optional[Dict[str, torch.Tensor]] = None
        self._k_float: Optional[torch.Tensor] = None
        self._v_float: Optional[torch.Tensor] = None
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.is_initialized = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seq_len={self.get_seq_length()})"

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def _decode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._compressed is None:
            z = torch.tensor([], dtype=self.dtype, device=self.device)
            return z, z
        k, v = self.quantizer.decompress_kv_cache(self._compressed)
        return k.to(dtype=self.dtype, device=self.device), v.to(dtype=self.dtype, device=self.device)

    def _materialized_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hybrid_float_cache and self._k_float is not None and self._k_float.numel() > 0:
            assert self._v_float is not None
            return self._k_float, self._v_float
        return self._decode()

    def _hybrid_append_float(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._k_float is None or self._k_float.numel() == 0:
            self._k_float = k_new
            self._v_float = v_new
        else:
            self._k_float = torch.cat([self._k_float, k_new], dim=-2)
            self._v_float = torch.cat([self._v_float, v_new], dim=-2)
        return self._k_float, self._v_float

    def _sync_hybrid_from_compressed(self) -> None:
        if not self.hybrid_float_cache:
            return
        if self._compressed is None:
            self._k_float = None
            self._v_float = None
            return
        k, v = self.quantizer.decompress_kv_cache(self._compressed)
        self._k_float = k.to(dtype=self.dtype, device=self.device)
        self._v_float = v.to(dtype=self.dtype, device=self.device)

    @property
    def keys(self) -> torch.Tensor:
        if not self.is_initialized or self._compressed is None:
            return torch.tensor([], dtype=self.dtype, device=self.device)
        k, _ = self._materialized_kv()
        return k

    @property
    def values(self) -> torch.Tensor:
        if not self.is_initialized or self._compressed is None:
            return torch.tensor([], dtype=self.dtype, device=self.device)
        _, v = self._materialized_kv()
        return v

    @property
    def compressed_kv(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._compressed

    @property
    def max_batch_size(self) -> int:
        if self._compressed is None:
            return 0
        return int(self._compressed["k_idx"].shape[0])

    @property
    def max_cache_len(self) -> int:
        return max(0, self.get_seq_length())

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs  # reserved
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        k_new = key_states.to(dtype=self.dtype, device=self.device)
        v_new = value_states.to(dtype=self.dtype, device=self.device)

        if self.strict_reencode:
            if self._compressed is None:
                k, v = k_new, v_new
            else:
                if self.hybrid_float_cache and self._k_float is not None and self._k_float.numel() > 0:
                    k_prev, v_prev = self._k_float, self._v_float
                else:
                    k_prev, v_prev = self._decode()
                if k_prev.numel() == 0:
                    k, v = k_new, v_new
                else:
                    k = torch.cat([k_prev, k_new], dim=-2)
                    v = torch.cat([v_prev, v_new], dim=-2)
            self._compressed = self.quantizer.quantize_kv(k, v, return_compressed=True)
        else:
            delta = self.quantizer.quantize_kv(k_new, v_new, return_compressed=True)
            self._compressed = concat_compressed_kv(self._compressed, delta)

        dev = k_new.device
        self._compressed = {kk: vv.to(dev) for kk, vv in self._compressed.items()}

        if self.hybrid_float_cache and not self.strict_reencode:
            k_out, v_out = self._hybrid_append_float(k_new, v_new)
        else:
            k_out, v_out = self.quantizer.decompress_kv_cache(self._compressed)
            k_out = k_out.to(dtype=self.dtype, device=self.device)
            v_out = v_out.to(dtype=self.dtype, device=self.device)
            if self.hybrid_float_cache and self.strict_reencode:
                self._k_float, self._v_float = k_out, v_out
        return k_out, v_out

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if not self.is_initialized or self._compressed is None:
            return 0
        return int(self._compressed["k_idx"].shape[2])

    def get_max_cache_shape(self) -> int:
        return -1

    def offload(self) -> None:
        if self.is_initialized and self._compressed is not None:
            self._compressed = {k: v.detach().to("cpu", non_blocking=True) for k, v in self._compressed.items()}
        if self.hybrid_float_cache and self._k_float is not None:
            assert self._v_float is not None
            self._k_float = self._k_float.detach().to("cpu", non_blocking=True)
            self._v_float = self._v_float.detach().to("cpu", non_blocking=True)
        self.device = torch.device("cpu")

    def prefetch(self) -> None:
        pass

    def reset(self) -> None:
        self._compressed = None
        self._k_float = None
        self._v_float = None
        self.is_initialized = False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self._compressed is None or self.get_seq_length() == 0:
            return
        k, v = self._decode()
        k = k.index_select(0, beam_idx.to(k.device))
        v = v.index_select(0, beam_idx.to(v.device))
        self._compressed = self.quantizer.quantize_kv(k, v, return_compressed=True)
        self._sync_hybrid_from_compressed()

    def crop(self, max_length: int) -> None:
        if self._compressed is None:
            return
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        k, v = self._decode()
        k = k[..., :max_length, :]
        v = v[..., :max_length, :]
        self._compressed = self.quantizer.quantize_kv(k, v, return_compressed=True)
        self._sync_hybrid_from_compressed()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self._compressed is None or self.get_seq_length() == 0:
            return
        k, v = self._decode()
        k = k.repeat_interleave(repeats, dim=0)
        v = v.repeat_interleave(repeats, dim=0)
        self._compressed = self.quantizer.quantize_kv(k, v, return_compressed=True)
        self._sync_hybrid_from_compressed()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self._compressed is None or self.get_seq_length() == 0:
            return
        k, v = self._decode()
        k = k[indices, ...]
        v = v[indices, ...]
        self._compressed = self.quantizer.quantize_kv(k, v, return_compressed=True)
        self._sync_hybrid_from_compressed()


class TurboQuantTritonFusedCacheLayer(TurboQuantCacheLayer):
    """
    Incremental compressed KV; :meth:`append_from_kv` for patched Llama (no full decode each step).

    With :func:`turboquant.hf_fused_attention.install_turboquant_fused_attention`, attention uses Triton fused
    on ``compressed_kv`` directly. Without the patch, ``update`` behaves like the base layer.
    """

    def append_from_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        del cache_kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        k_new = key_states.to(dtype=self.dtype, device=self.device)
        v_new = value_states.to(dtype=self.dtype, device=self.device)
        delta = self.quantizer.quantize_kv(k_new, v_new, return_compressed=True)
        self._compressed = concat_compressed_kv(self._compressed, delta)
        dev = k_new.device
        self._compressed = {kk: vv.to(dev) for kk, vv in self._compressed.items()}
        if self.hybrid_float_cache:
            self._hybrid_append_float(k_new, v_new)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.append_from_kv(key_states, value_states, cache_kwargs)
        if self.hybrid_float_cache:
            assert self._k_float is not None and self._v_float is not None
            return self._k_float, self._v_float
        return self._decode()


class TurboQuantDynamicCache(DynamicCache):
    """
    Same layer layout inference as ``DynamicCache(config=...)``, but full-attention layers use
    ``TurboQuantCacheLayer`` or ``TurboQuantTritonFusedCacheLayer``.
    """

    def __init__(
        self,
        config: "PreTrainedConfig",
        quantizer: TurboQuantProd,
        *,
        triton_fused_layers: bool = False,
        strict_reencode: bool = False,
        hybrid_float_cache: bool = False,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        from transformers.cache_utils import DynamicSlidingWindowLayer

        layers: list = []
        decoder_config = _text_decoder_config(config)
        sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
            decoder_config, "attention_chunk_size", None
        )
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "sliding_attention" if sliding_window is not None else "full_attention"
                for _ in range(decoder_config.num_hidden_layers)
            ]
        if hasattr(decoder_config, "num_kv_shared_layers"):
            layer_types = layer_types[: -int(decoder_config.num_kv_shared_layers)]

        layer_kw = {"strict_reencode": strict_reencode, "hybrid_float_cache": hybrid_float_cache}
        for layer_type in layer_types:
            if layer_type in ("sliding_attention", "chunked_attention"):
                if sliding_window is None:
                    # Misconfigured configs can mark sliding layers without a window size; mirror "full" path.
                    if triton_fused_layers:
                        layers.append(TurboQuantTritonFusedCacheLayer(quantizer, **layer_kw))
                    else:
                        layers.append(TurboQuantCacheLayer(quantizer, **layer_kw))
                else:
                    layers.append(DynamicSlidingWindowLayer(sliding_window=int(sliding_window)))
            elif triton_fused_layers:
                layers.append(TurboQuantTritonFusedCacheLayer(quantizer, **layer_kw))
            else:
                layers.append(TurboQuantCacheLayer(quantizer, **layer_kw))

        Cache.__init__(self, layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)


def turboquant_dynamic_cache(
    config: "PreTrainedConfig",
    quantizer: TurboQuantProd,
    *,
    triton_fused_layers: bool = False,
    strict_reencode: bool = False,
    hybrid_float_cache: bool = False,
    offloading: bool = False,
    offload_only_non_sliding: bool = False,
) -> "TurboQuantDynamicCache":
    """
    Factory: ``past_key_values=turboquant_dynamic_cache(model.config, quantizer)``.

    ``triton_fused_layers=True`` builds :class:`TurboQuantTritonFusedCacheLayer` for full-attention
    layers; use with :func:`turboquant.hf_fused_attention.install_turboquant_fused_attention`.
    """
    return TurboQuantDynamicCache(
        config,
        quantizer,
        triton_fused_layers=triton_fused_layers,
        strict_reencode=strict_reencode,
        hybrid_float_cache=hybrid_float_cache,
        offloading=offloading,
        offload_only_non_sliding=offload_only_non_sliding,
    )


def export_compressed_to_paged(
    compressed_kv: Dict[str, torch.Tensor],
    block_size: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    return pack_dense_kv_to_paged(compressed_kv, block_size)


def export_cache_to_paged_per_layer(
    cache: Any,
    block_size: int,
) -> List[Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]]:
    out: List[Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]] = []
    for layer in getattr(cache, "layers", []):
        if isinstance(layer, TurboQuantCacheLayer):  # includes TurboQuantTritonFusedCacheLayer
            ck = layer.compressed_kv
            if ck is None:
                out.append(None)
            else:
                out.append(export_compressed_to_paged(ck, block_size))
        else:
            out.append(None)
    return out


VLLM_INTEGRATION_NOTES = """
vLLM uses its own paged KV allocator and ``cache_dtype`` options (fp8, bf16, …). TurboQuant adds
a separate ``turboquant`` mode: a new AttentionBackend, ``TurboQuantAttentionSpec`` (page = fixed-size
uint8 row) and the same Triton fused decode as in ``turboquant.kernels.fused_attention``.

Patch for the vLLM tree, apply script, and overlay: ``integrations/vllm_upstream/patches/``,
``integrations/vllm_upstream/apply_to_vllm.py``, ``integrations/vllm_upstream/README.md``,
manual port checklist — ``integrations/vllm_upstream/UPSTREAM_EDITS.md``; layout —
``turboquant.vllm_pack``.

KV writes still go through Python (scatter by ``slot_mapping``); a native op at the level of
``reshape_and_cache_flash`` is a separate step for a vLLM PR.

Our ``export_cache_to_paged_per_layer`` is compatible with ``quantized_attention_fused_triton_paged``; after
vLLM integration the internal paged buffer is of the same family as ``uint8_pages_to_paged_dict``.
"""

LLAMA_CPP_INTEGRATION_NOTES = """
**llama.cpp:** binary sidecar ``*.tqmeta`` (Π, S, centroids, ``qjl_factor``) —
``turboquant.llama_cpp_pack``; KV page layout matches vLLM: ``turboquant.vllm_pack``.
Guide and reference C++ CPU dequant: ``integrations/llama_cpp/README.md``,
``integrations/llama_cpp/reference/turboquant_sidecar_loader.cpp``,
upstream edit checklist: ``integrations/llama_cpp/UPSTREAM_EDITS.md``.
"""
