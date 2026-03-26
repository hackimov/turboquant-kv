"""
Multi-architecture **decoder** attention with fused compressed-KV path on :class:`TurboQuantTritonFusedCacheLayer`.

**Parity with HF eager** (within TurboQuant score approximation) when all of the following hold:

- Stock layout: ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``, RoPE via that model’s ``apply_rotary_pos_emb``,
  ``layer_idx``, ``config.num_key_value_heads``, ``head_dim``.
- Logits are ``score * (head_dim ** -0.5)`` — same scaling as our Triton kernel. If the config uses a different
  query scale (e.g. Gemma2 ``query_pre_attn_scalar``) or **attention logit softcapping** (Gemma2
  ``attn_logit_softcapping``), this module **automatically** uses the stock ``forward`` so results match HF.
- Dense additive ``attention_mask`` (see module doc in earlier paragraphs); flex / ``BlockMask`` → stock ``forward``.
- ``TurboQuantTritonFusedCacheLayer`` on that layer (``turboquant_dynamic_cache(..., triton_fused_layers=True)`` and
  a **full_attention** cache slot — sliding-window HF layers stay stock). Backend: Triton on CUDA, SDPA fallback
  on Apple Metal (MPS).

**Not** covered in fused form for **generic** wrappers (use stock attention + quant cache or extend upstream): fused QKV without a dedicated path (e.g. GPT-NeoX). **Phi-4 Multimodal** text attention uses fused QKV but has a **dedicated** TurboQuant wrapper (``phi4_multimodal``).
query/key head RMSNorms beyond the Qwen3-style path (e.g. Olmo2), MoE routing, native sliding-window **compressed** KV.

Install: :func:`install_turboquant_fused_attention` (alias :func:`install_decoder_fused_attention`).
Llama-specific names: ``turboquant.hf_llama_fused``.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from .core import TurboQuantProd
from .hf_cache import TurboQuantTritonFusedCacheLayer
from .kernels.attention_mask import broadcast_additive_attn_mask


def triton_cuda_available() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def fused_attention_backend_available(hidden_states: torch.Tensor) -> bool:
    # Triton fused kernels on CUDA, or SDPA fallback path on Apple Metal (MPS).
    return (hidden_states.is_cuda and triton_cuda_available()) or (hidden_states.device.type == "mps")


def _inner_decoder_stack(model: nn.Module) -> nn.Module:
    inner = model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model
    if not hasattr(inner, "layers"):
        raise TypeError("Expected a decoder-only HF model with .model.layers or .layers")
    return inner


def _resolve_fused_additive_mask(
    attention_mask: Optional[Any],
    *,
    Bq: int,
    Hq: int,
    M: int,
    N: int,
    cache_position: Optional[torch.LongTensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        return None

    if attention_mask is not None:
        if attention_mask.dim() != 4:
            return None
        if attention_mask.shape[-2] != M or attention_mask.shape[-1] != N:
            return None
        try:
            return broadcast_additive_attn_mask(attention_mask, Bq, Hq, M, N, device=device)
        except ValueError:
            return None

    if cache_position is None or cache_position.numel() != M:
        return None
    cp = cache_position.to(device=device, dtype=torch.long)
    ar_n = torch.arange(N, device=device, dtype=torch.long).view(1, 1, 1, N)
    ar_m = cp.view(1, 1, M, 1)
    z = torch.zeros((), device=device, dtype=torch.float32)
    neg = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    additive = torch.where(ar_n <= ar_m, z, neg)
    return additive.expand(Bq, Hq, M, N)


def _attention_requires_stock_hf_forward(module: nn.Module) -> bool:
    """
    True when HF eager applies extra logits transforms our fused kernel does not implement identically.

    - Attention logit softcapping (``tanh`` pipeline), used by Gemma2.
    - Query scaling not tied to ``head_dim ** -0.5`` (e.g. Gemma2 ``query_pre_attn_scalar``).
    """
    cfg = getattr(module, "config", None)
    if cfg is None:
        return False
    soft = getattr(cfg, "attn_logit_softcapping", None)
    if soft is not None:
        try:
            if float(soft) != 0.0:
                return True
        except (TypeError, ValueError):
            return True
    hd = int(getattr(module, "head_dim", 0) or 0)
    if hd <= 0:
        return False
    std_scale = float(hd) ** -0.5
    qpas = getattr(cfg, "query_pre_attn_scalar", None)
    if qpas is not None:
        try:
            alt = float(qpas) ** -0.5
            if abs(alt - std_scale) > 1e-5:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _resolve_registered_attention_base(
    typ: Type[nn.Module], reg: Dict[Type[nn.Module], Type[nn.Module]]
) -> Optional[Type[nn.Module]]:
    for cls in typ.__mro__:
        if cls in reg:
            return cls
    return None


def _turboquant_fused_attention_forward(
    self: nn.Module,
    super_forward: Callable[..., Any],
    apply_rotary_pos_emb: Callable[..., Any],
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Any,
    cache_position: Optional[torch.LongTensor],
    **kwargs: Any,
) -> Tuple[torch.Tensor, Any]:
    quantizer: Optional[TurboQuantProd] = getattr(self, "_turboquant_quantizer", None)

    can_fused = (
        quantizer is not None
        and fused_attention_backend_available(hidden_states)
        and past_key_values is not None
        and self.layer_idx < len(past_key_values.layers)
        and isinstance(past_key_values.layers[self.layer_idx], TurboQuantTritonFusedCacheLayer)
        and int(quantizer.head_dim) == int(self.head_dim)
    )

    if not can_fused:
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    if _attention_requires_stock_hf_forward(self):
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qh = self.q_proj(hidden_states).view(hidden_shape)
    kh = self.k_proj(hidden_states).view(hidden_shape)
    vh = self.v_proj(hidden_states).view(hidden_shape)
    qn = getattr(self, "q_norm", None)
    kn = getattr(self, "k_norm", None)
    if qn is not None:
        qh = qn(qh)
    if kn is not None:
        kh = kn(kh)
    query_states = qh.transpose(1, 2)
    key_states = kh.transpose(1, 2)
    value_states = vh.transpose(1, 2)

    if position_embeddings is None:
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_layer = past_key_values.layers[self.layer_idx]
    M = int(query_states.shape[2])
    seq_before = cache_layer.get_seq_length()
    N_expected = seq_before + M
    Bq, Hq = query_states.shape[0], query_states.shape[1]

    mask_4d = _resolve_fused_additive_mask(
        attention_mask,
        Bq=Bq,
        Hq=Hq,
        M=M,
        N=N_expected,
        cache_position=cache_position,
        device=query_states.device,
    )
    if mask_4d is None:
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    cache_layer.append_from_kv(key_states, value_states, cache_kwargs)

    kv = cache_layer.compressed_kv
    if kv is None or int(kv["k_idx"].shape[2]) != N_expected:
        raise RuntimeError(
            "TurboQuant fused cache invariant broken after append_from_kv "
            f"(got seq {0 if kv is None else kv['k_idx'].shape[2]}, expected {N_expected})."
        )

    attn_out = quantizer.quantized_attention_fused_auto(
        query_states,
        kv,
        num_kv_heads=int(self.config.num_key_value_heads),
        causal=False,
        attention_mask=mask_4d,
    )
    attn_out = attn_out.transpose(1, 2).contiguous()
    attn_out = attn_out.reshape(*input_shape, -1).contiguous()
    attn_out = self.o_proj(attn_out)
    return attn_out, None


def _phi4mm_sliding_window_config_active(module: nn.Module) -> bool:
    """HF passes ``sliding_window`` into attention; our Triton fused kernel does not implement that mask."""
    cfg = getattr(module, "config", None)
    if cfg is None:
        return False
    sw = getattr(cfg, "sliding_window", None)
    if sw is None:
        return False
    try:
        return int(sw) > 0
    except (TypeError, ValueError):
        return True


def _turboquant_phi4_multimodal_attention_forward(
    self: nn.Module,
    super_forward: Callable[..., Any],
    apply_rotary_pos_emb: Callable[..., Any],
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Any,
    cache_position: Optional[torch.LongTensor],
    **kwargs: Any,
) -> Tuple[torch.Tensor, Any]:
    """
    Same contract as :func:`_turboquant_fused_attention_forward`, but Q/K/V from ``qkv_proj`` (Phi-4 Multimodal text).
    """
    quantizer: Optional[TurboQuantProd] = getattr(self, "_turboquant_quantizer", None)

    can_fused = (
        quantizer is not None
        and fused_attention_backend_available(hidden_states)
        and past_key_values is not None
        and getattr(self, "layer_idx", None) is not None
        and int(self.layer_idx) < len(past_key_values.layers)
        and isinstance(past_key_values.layers[int(self.layer_idx)], TurboQuantTritonFusedCacheLayer)
        and int(quantizer.head_dim) == int(self.head_dim)
    )

    if not can_fused:
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    if _attention_requires_stock_hf_forward(self) or _phi4mm_sliding_window_config_active(self):
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qkv = self.qkv_proj(hidden_states)
    query_pos = int(self.config.num_attention_heads) * int(self.head_dim)
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + int(self.num_key_value_heads) * int(self.head_dim)]
    value_states = qkv[..., query_pos + int(self.num_key_value_heads) * int(self.head_dim) :]

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    if position_embeddings is None:
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_layer = past_key_values.layers[int(self.layer_idx)]
    M = int(query_states.shape[2])
    seq_before = cache_layer.get_seq_length()
    N_expected = seq_before + M
    Bq, Hq = query_states.shape[0], query_states.shape[1]

    mask_4d = _resolve_fused_additive_mask(
        attention_mask,
        Bq=Bq,
        Hq=Hq,
        M=M,
        N=N_expected,
        cache_position=cache_position,
        device=query_states.device,
    )
    if mask_4d is None:
        return super_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    cache_layer.append_from_kv(key_states, value_states, cache_kwargs)

    kv = cache_layer.compressed_kv
    if kv is None or int(kv["k_idx"].shape[2]) != N_expected:
        raise RuntimeError(
            "TurboQuant fused cache invariant broken after append_from_kv "
            f"(got seq {0 if kv is None else kv['k_idx'].shape[2]}, expected {N_expected})."
        )

    attn_out = quantizer.quantized_attention_fused_auto(
        query_states,
        kv,
        num_kv_heads=int(self.config.num_key_value_heads),
        causal=False,
        attention_mask=mask_4d,
    )
    attn_out = attn_out.transpose(1, 2).contiguous()
    attn_out = attn_out.reshape(*input_shape, -1).contiguous()
    attn_out = self.o_proj(attn_out)
    return attn_out, None


# --- Per-architecture wrappers (explicit classes for correct MRO / state_dict) ---


def _make_wrapper(
    base: Type[nn.Module],
    apply_rotary_mod: str,
    apply_rotary_name: str = "apply_rotary_pos_emb",
    class_name: str = "TurboQuantAttention",
) -> Type[nn.Module]:
    apply_rotary = getattr(__import__(apply_rotary_mod, fromlist=[apply_rotary_name]), apply_rotary_name)

    class _W(base):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._turboquant_quantizer: Optional[TurboQuantProd] = None

        def bind_turboquant(self, quantizer: TurboQuantProd) -> "_W":
            self._turboquant_quantizer = quantizer
            return self

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Any = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Any]:
            def _super(
                hs: torch.Tensor,
                pe: Optional[Tuple[torch.Tensor, torch.Tensor]],
                am: Optional[torch.Tensor] = None,
                pk: Any = None,
                cp: Optional[torch.LongTensor] = None,
                **kw: Any,
            ) -> Any:
                return super(_W, self).forward(hs, pe, am, pk, cp, **kw)

            return _turboquant_fused_attention_forward(
                self,
                _super,
                apply_rotary,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

    _W.__name__ = _W.__qualname__ = class_name
    return _W


def _make_phi4_multimodal_wrapper(
    base: Type[nn.Module],
    apply_rotary: Callable[..., Any],
    class_name: str = "TurboQuantPhi4MultimodalAttention",
) -> Type[nn.Module]:
    class _W(base):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._turboquant_quantizer: Optional[TurboQuantProd] = None

        def bind_turboquant(self, quantizer: TurboQuantProd) -> "_W":
            self._turboquant_quantizer = quantizer
            return self

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Any = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Any]:
            def _super(
                hs: torch.Tensor,
                pe: Optional[Tuple[torch.Tensor, torch.Tensor]],
                am: Optional[torch.Tensor] = None,
                pk: Any = None,
                cp: Optional[torch.LongTensor] = None,
                **kw: Any,
            ) -> Any:
                return super(_W, self).forward(hs, pe, am, pk, cp, **kw)

            return _turboquant_phi4_multimodal_attention_forward(
                self,
                _super,
                apply_rotary,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

    _W.__name__ = _W.__qualname__ = class_name
    return _W


def _load_registry() -> Tuple[Dict[Type[nn.Module], Type[nn.Module]], Dict[str, Type[nn.Module]], List[Type[nn.Module]]]:
    """Returns base->wrapper, arch alias -> HF base class, list of wrapper classes."""
    reg: Dict[Type[nn.Module], Type[nn.Module]] = {}
    aliases: Dict[str, Type[nn.Module]] = {}
    wrappers: List[Type[nn.Module]] = []

    def _add(alias: str, base_mod: str, base_cls: str, rope_mod: str, wrap_name: str) -> None:
        try:
            bmod = __import__(base_mod, fromlist=[base_cls])
            Base = getattr(bmod, base_cls)
        except (ImportError, AttributeError):
            return
        Wrap = _make_wrapper(Base, rope_mod, class_name=wrap_name)
        reg[Base] = Wrap
        aliases[alias] = Base
        wrappers.append(Wrap)

    _add("llama", "transformers.models.llama.modeling_llama", "LlamaAttention", "transformers.models.llama.modeling_llama", "TurboQuantLlamaAttention")
    _add("mistral", "transformers.models.mistral.modeling_mistral", "MistralAttention", "transformers.models.mistral.modeling_mistral", "TurboQuantMistralAttention")
    _add("qwen2", "transformers.models.qwen2.modeling_qwen2", "Qwen2Attention", "transformers.models.qwen2.modeling_qwen2", "TurboQuantQwen2Attention")
    _add("qwen3", "transformers.models.qwen3.modeling_qwen3", "Qwen3Attention", "transformers.models.qwen3.modeling_qwen3", "TurboQuantQwen3Attention")
    _add("gemma2", "transformers.models.gemma2.modeling_gemma2", "Gemma2Attention", "transformers.models.gemma2.modeling_gemma2", "TurboQuantGemma2Attention")
    _add("phi3", "transformers.models.phi3.modeling_phi3", "Phi3Attention", "transformers.models.phi3.modeling_phi3", "TurboQuantPhi3Attention")
    _add("cohere", "transformers.models.cohere.modeling_cohere", "CohereAttention", "transformers.models.cohere.modeling_cohere", "TurboQuantCohereAttention")
    _add("granite", "transformers.models.granite.modeling_granite", "GraniteAttention", "transformers.models.granite.modeling_granite", "TurboQuantGraniteAttention")
    _add(
        "starcoder2",
        "transformers.models.starcoder2.modeling_starcoder2",
        "Starcoder2Attention",
        "transformers.models.starcoder2.modeling_starcoder2",
        "TurboQuantStarcoder2Attention",
    )

    # Phi-4 / Phi-4-mini (Microsoft Hub) use ``Phi3Attention`` and ``model_type="phi3"``; explicit ``phi4`` alias.
    if "phi3" in aliases:
        aliases["phi4"] = aliases["phi3"]

    def _add_phi4_multimodal() -> None:
        try:
            bmod = __import__(
                "transformers.models.phi4_multimodal.modeling_phi4_multimodal",
                fromlist=["Phi4MultimodalAttention", "apply_rotary_pos_emb"],
            )
            Base = bmod.Phi4MultimodalAttention
            apply_rotary = bmod.apply_rotary_pos_emb
        except (ImportError, AttributeError):
            return
        Wrap = _make_phi4_multimodal_wrapper(Base, apply_rotary)
        reg[Base] = Wrap
        aliases["phi4_multimodal"] = Base
        wrappers.append(Wrap)

    _add_phi4_multimodal()

    return reg, aliases, wrappers


_REGISTRY: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None
_ALIASES: Optional[Dict[str, Type[nn.Module]]] = None
_WRAPPER_TYPES: Optional[List[Type[nn.Module]]] = None


def _get_registry() -> Tuple[Dict[Type[nn.Module], Type[nn.Module]], Dict[str, Type[nn.Module]], List[Type[nn.Module]]]:
    global _REGISTRY, _ALIASES, _WRAPPER_TYPES
    if _REGISTRY is None:
        _REGISTRY, _ALIASES, _WRAPPER_TYPES = _load_registry()
    assert _REGISTRY is not None and _ALIASES is not None and _WRAPPER_TYPES is not None
    return _REGISTRY, _ALIASES, _WRAPPER_TYPES


def supported_fused_attention_architectures() -> List[str]:
    """
    Lower-case names accepted by ``install_turboquant_fused_attention(..., architecture=...)``.

    Encoder-decoder stacks (**M2M-100**, **NLLB** checkpoints that use ``model_type: m2m_100``, **MarianMT**,
    **T5**, …) are **not** listed here: their attention modules differ from the decoder-only registry below.
    Use :func:`turboquant.hf_cache.turboquant_encoder_decoder_cache` for **cross-attention KV** compression.

    ``phi4`` is an alias for the same HF base as ``phi3`` (Phi-4 / Phi-4-mini use ``Phi3Attention``).
    ``phi4_multimodal`` is **Phi-4 Multimodal** text attention (``Phi4MultimodalAttention``, fused QKV).
    ``internlm2`` / ``internlm3`` use Hub remote-code modules (see ``turboquant.hf_internlm_fused``).
    """
    _, aliases, _ = _get_registry()
    names = set(aliases.keys())
    names.update(("internlm2", "internlm3"))
    return sorted(names)


def install_turboquant_fused_attention(
    model: nn.Module,
    quantizer: TurboQuantProd,
    *,
    architecture: str = "auto",
    allow_attention_subclass: bool = False,
) -> None:
    """
    Replace each ``layer.self_attn`` with the TurboQuant wrapper for the given HF attention class.

    ``architecture="auto"``: infer the stock base class from ``type(layer.self_attn)`` (exact type by default, or
    first match in the MRO against the registry if ``allow_attention_subclass=True``).

    ``allow_attention_subclass``: if True, allow custom subclasses of a registered base (e.g. ``class T(LlamaAttention)``);
    swap uses ``RegisteredBase(config, layer_idx)`` and ``load_state_dict`` from your module (extra parameters must
    not break strict loading).
    """
    reg, aliases, wrappers = _get_registry()
    inner = _inner_decoder_stack(model)
    arch_l = architecture.strip().lower()

    if arch_l in ("internlm2", "internlm3"):
        from .hf_internlm_fused import install_internlm_decoder_fused_attention

        return install_internlm_decoder_fused_attention(
            model,
            quantizer,
            architecture=arch_l,
            allow_attention_subclass=allow_attention_subclass,
        )

    if arch_l == "auto":
        sample_name = type(inner.layers[0].self_attn).__name__
        if sample_name.startswith("InternLM2") and "Attention" in sample_name:
            from .hf_internlm_fused import install_internlm_decoder_fused_attention

            return install_internlm_decoder_fused_attention(
                model,
                quantizer,
                architecture="internlm2",
                allow_attention_subclass=allow_attention_subclass,
            )
        if sample_name.startswith("InternLM3") and "Attention" in sample_name:
            from .hf_internlm_fused import install_internlm_decoder_fused_attention

            return install_internlm_decoder_fused_attention(
                model,
                quantizer,
                architecture="internlm3",
                allow_attention_subclass=allow_attention_subclass,
            )

    expected_base: Optional[Type[nn.Module]] = None
    if arch_l != "auto":
        if arch_l not in aliases:
            raise ValueError(
                f"Unknown architecture {architecture!r}. Choose one of: auto, {', '.join(supported_fused_attention_architectures())}"
            )
        expected_base = aliases[arch_l]

    for layer in inner.layers:
        cur = layer.self_attn
        if any(isinstance(cur, w) for w in wrappers):
            cur.bind_turboquant(quantizer)
            continue

        cur_type = type(cur)
        wrap_key: Optional[Type[nn.Module]] = None

        if expected_base is not None:
            if not isinstance(cur, expected_base):
                raise TypeError(
                    f"architecture={architecture!r} requires isinstance(..., {expected_base.__name__}), "
                    f"got {cur_type.__name__}"
                )
            if allow_attention_subclass:
                mro_base = _resolve_registered_attention_base(cur_type, reg)
                if mro_base is not expected_base:
                    raise TypeError(
                        f"architecture={architecture!r} expects registry base {expected_base.__name__}, "
                        f"but MRO resolved {mro_base.__name__ if mro_base else None}"
                    )
                wrap_key = expected_base
            else:
                if cur_type is not expected_base:
                    raise TypeError(
                        f"architecture={architecture!r} requires exact type {expected_base.__name__}, "
                        f"got {cur_type.__name__} (try allow_attention_subclass=True)"
                    )
                wrap_key = expected_base
        else:
            if allow_attention_subclass:
                wrap_key = _resolve_registered_attention_base(cur_type, reg)
                if wrap_key is None:
                    raise TypeError(
                        f"Unsupported attention module {cur_type.__name__}: no registered base in MRO. "
                        f"Known bases: {', '.join(sorted(b.__name__ for b in reg))}."
                    )
            else:
                if cur_type not in reg:
                    raise TypeError(
                        f"Unsupported attention module {cur_type.__name__} (exact type not registered). "
                        f"Supported: {', '.join(sorted(b.__name__ for b in reg))}. "
                        f"Try allow_attention_subclass=True for subclasses."
                    )
                wrap_key = cur_type

        assert wrap_key is not None
        Wrap = reg[wrap_key]
        new = Wrap(cur.config, layer_idx=cur.layer_idx)
        new.load_state_dict(cur.state_dict(), strict=True)
        dev = next(cur.parameters()).device
        dt = next(cur.parameters()).dtype
        new.to(device=dev, dtype=dt)
        new.bind_turboquant(quantizer)
        layer.self_attn = new


def uninstall_turboquant_fused_attention(model: nn.Module) -> None:
    """Restore stock HF attention modules for any TurboQuant wrapper known to this module."""
    from .hf_internlm_fused import uninstall_internlm_decoder_fused_attention

    uninstall_internlm_decoder_fused_attention(model)

    reg, _, wrappers = _get_registry()
    inner = _inner_decoder_stack(model)
    base_by_wrap = {w: b for b, w in reg.items()}

    for layer in inner.layers:
        cur = layer.self_attn
        for w in wrappers:
            if isinstance(cur, w):
                Base = base_by_wrap[w]
                restored = Base(cur.config, layer_idx=cur.layer_idx)
                restored.load_state_dict(cur.state_dict(), strict=True)
                dev = next(cur.parameters()).device
                dt = next(cur.parameters()).dtype
                restored.to(device=dev, dtype=dt)
                layer.self_attn = restored
                break


install_decoder_fused_attention = install_turboquant_fused_attention
uninstall_decoder_fused_attention = uninstall_turboquant_fused_attention


def _export_attention_classes() -> None:
    """Populate module-level ``TurboQuant*Attention`` classes from the registry."""
    global TurboQuantLlamaAttention, TurboQuantMistralAttention, TurboQuantQwen2Attention, TurboQuantQwen3Attention
    global TurboQuantGemma2Attention, TurboQuantPhi3Attention, TurboQuantPhi4Attention
    global TurboQuantPhi4MultimodalAttention, TurboQuantCohereAttention
    global TurboQuantGraniteAttention, TurboQuantStarcoder2Attention

    reg, _, _ = _get_registry()
    pairs = [
        ("TurboQuantLlamaAttention", "transformers.models.llama.modeling_llama", "LlamaAttention"),
        ("TurboQuantMistralAttention", "transformers.models.mistral.modeling_mistral", "MistralAttention"),
        ("TurboQuantQwen2Attention", "transformers.models.qwen2.modeling_qwen2", "Qwen2Attention"),
        ("TurboQuantQwen3Attention", "transformers.models.qwen3.modeling_qwen3", "Qwen3Attention"),
        ("TurboQuantGemma2Attention", "transformers.models.gemma2.modeling_gemma2", "Gemma2Attention"),
        ("TurboQuantPhi3Attention", "transformers.models.phi3.modeling_phi3", "Phi3Attention"),
        ("TurboQuantPhi4Attention", "transformers.models.phi3.modeling_phi3", "Phi3Attention"),
        ("TurboQuantPhi4MultimodalAttention", "transformers.models.phi4_multimodal.modeling_phi4_multimodal", "Phi4MultimodalAttention"),
        ("TurboQuantCohereAttention", "transformers.models.cohere.modeling_cohere", "CohereAttention"),
        ("TurboQuantGraniteAttention", "transformers.models.granite.modeling_granite", "GraniteAttention"),
        ("TurboQuantStarcoder2Attention", "transformers.models.starcoder2.modeling_starcoder2", "Starcoder2Attention"),
    ]
    g = globals()
    for attr, mod, cls_name in pairs:
        try:
            bmod = __import__(mod, fromlist=[cls_name])
            Base = getattr(bmod, cls_name)
            g[attr] = reg[Base]
        except (ImportError, AttributeError, KeyError):
            g[attr] = None


TurboQuantLlamaAttention = None  # type: ignore[misc, assignment]
TurboQuantMistralAttention = None  # type: ignore[misc, assignment]
TurboQuantQwen2Attention = None  # type: ignore[misc, assignment]
TurboQuantQwen3Attention = None  # type: ignore[misc, assignment]
TurboQuantGemma2Attention = None  # type: ignore[misc, assignment]
TurboQuantPhi3Attention = None  # type: ignore[misc, assignment]
TurboQuantPhi4Attention = None  # type: ignore[misc, assignment]
TurboQuantPhi4MultimodalAttention = None  # type: ignore[misc, assignment]
TurboQuantCohereAttention = None  # type: ignore[misc, assignment]
TurboQuantGraniteAttention = None  # type: ignore[misc, assignment]
TurboQuantStarcoder2Attention = None  # type: ignore[misc, assignment]

_export_attention_classes()
del _export_attention_classes
