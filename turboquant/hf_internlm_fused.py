"""
InternLM2 / InternLM2.5 / InternLM3 decoder attention for TurboQuant fused KV.

These models ship **remote code** on the Hugging Face Hub (not in ``transformers`` core). We resolve
``InternLM2Attention`` / ``InternLM3Attention`` from ``layer.self_attn.__class__.__module__`` after
``from_pretrained(..., trust_remote_code=True)``.

- **InternLM2**: fused ``wqkv`` + per-layer ``rotary_emb`` (``position_ids``), ``wo`` output projection.
- **InternLM3**: separate ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``, Llama-like ``position_embeddings`` optional.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from .core import TurboQuantProd
from .hf_cache import TurboQuantTritonFusedCacheLayer
from .hf_fused_attention import (
    _attention_requires_stock_hf_forward,
    _inner_decoder_stack,
    _resolve_fused_additive_mask,
    _resolve_registered_attention_base,
    fused_attention_backend_available,
)

# Wrapper types created for Hub modules (uninstall must recognize them).
_INTERNLM_WRAPPERS: List[Type[nn.Module]] = []
_WRAPPER_CACHE: Dict[Tuple[str, str], Type[nn.Module]] = {}


def _internlm2_qkv(
    self: nn.Module,
    hidden_states: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Returns (query, key, value) states [B, H, T, D] or ``None`` to fall back to stock forward."""
    if int(getattr(self.config, "pretraining_tp", 1) or 1) > 1:
        return None
    bsz, q_len, _ = hidden_states.shape
    qkv = self.wqkv(hidden_states)
    h = int(self.num_key_value_heads)
    gs = 2 + int(self.num_key_value_groups)
    d = int(self.head_dim)
    qkv_states = qkv.view(bsz, q_len, h, gs, d)
    q_part = qkv_states[:, :, :, : int(self.num_key_value_groups), :]
    query_states = q_part.reshape(bsz, q_len, -1, d).transpose(1, 2)
    key_states = qkv_states[:, :, :, -2, :].transpose(1, 2)
    value_states = qkv_states[:, :, :, -1, :].transpose(1, 2)
    return query_states, key_states, value_states


def _turboquant_internlm2_attention_forward(
    self: nn.Module,
    super_forward: Callable[..., Any],
    apply_rotary_pos_emb: Callable[..., Any],
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_value: Any,
    output_attentions: bool,
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, Any, Any]:
    quantizer: Optional[TurboQuantProd] = getattr(self, "_turboquant_quantizer", None)

    can_fused = (
        quantizer is not None
        and fused_attention_backend_available(hidden_states)
        and past_key_value is not None
        and self.layer_idx is not None
        and int(self.layer_idx) < len(past_key_value.layers)
        and isinstance(past_key_value.layers[int(self.layer_idx)], TurboQuantTritonFusedCacheLayer)
        and int(quantizer.head_dim) == int(self.head_dim)
    )

    if not can_fused:
        return super_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )

    if _attention_requires_stock_hf_forward(self):
        return super_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )

    qkv = _internlm2_qkv(self, hidden_states)
    if qkv is None:
        return super_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )
    query_states, key_states, value_states = qkv

    if position_ids is None:
        return super_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    cache_layer = past_key_value.layers[int(self.layer_idx)]
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
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
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
    bsz, q_len, _ = hidden_states.shape
    attn_out = attn_out.reshape(bsz, q_len, -1).contiguous()
    attn_out = self.wo(attn_out)
    attn_weights = None if not output_attentions else None
    return attn_out, attn_weights, past_key_value


def _make_internlm2_wrapper(base: Type[nn.Module], apply_rotary: Callable[..., Any]) -> Type[nn.Module]:
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
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Any = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Any, Any]:
            def _super(
                hs: torch.Tensor,
                am: Optional[torch.Tensor],
                pid: Optional[torch.LongTensor],
                pk: Any,
                oa: bool,
                uc: bool,
                cp: Optional[torch.LongTensor],
            ) -> Any:
                return super(_W, self).forward(hs, am, pid, pk, oa, uc, cp)

            return _turboquant_internlm2_attention_forward(
                self,
                _super,
                apply_rotary,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
            )

    _W.__name__ = _W.__qualname__ = "TurboQuantInternLM2Attention"
    return _W


def _make_internlm3_wrapper(base: Type[nn.Module], apply_rotary: Callable[..., Any]) -> Type[nn.Module]:
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
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Any = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Any, Any]:
            from .hf_fused_attention import _turboquant_fused_attention_forward

            def _super(
                hs: torch.Tensor,
                pe: Optional[Tuple[torch.Tensor, torch.Tensor]],
                am: Optional[torch.Tensor],
                pk: Any,
                cp: Optional[torch.LongTensor],
                **kw: Any,
            ) -> Any:
                return super(_W, self).forward(
                    hidden_states=hs,
                    attention_mask=am,
                    position_ids=kw.get("position_ids", position_ids),
                    past_key_value=pk,
                    output_attentions=kw.get("output_attentions", output_attentions),
                    use_cache=kw.get("use_cache", use_cache),
                    cache_position=cp,
                    position_embeddings=pe,
                    **{k: v for k, v in kw.items() if k not in ("position_ids", "output_attentions", "use_cache")},
                )

            out = _turboquant_fused_attention_forward(
                self,
                _super,
                apply_rotary,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1], past_key_value
            return out

    _W.__name__ = _W.__qualname__ = "TurboQuantInternLM3Attention"
    return _W


def _resolve_base_and_module(model: nn.Module, arch: str) -> Tuple[Type[nn.Module], Any]:
    attn0 = _inner_decoder_stack(model).layers[0].self_attn
    mod = importlib.import_module(attn0.__class__.__module__)
    if arch == "internlm2":
        Base = getattr(mod, "InternLM2Attention", None)
    else:
        Base = getattr(mod, "InternLM3Attention", None)
    if Base is None:
        raise ImportError(
            f"Could not find {'InternLM2Attention' if arch == 'internlm2' else 'InternLM3Attention'} "
            f"in module {mod.__name__!r}. Load the model with trust_remote_code=True."
        )
    return Base, mod


def _get_or_create_wrapper(base: Type[nn.Module], arch: str, modeling_mod: Any) -> Type[nn.Module]:
    key = (modeling_mod.__name__, arch)
    if key in _WRAPPER_CACHE:
        return _WRAPPER_CACHE[key]
    apply_rotary = getattr(modeling_mod, "apply_rotary_pos_emb")
    if arch == "internlm2":
        W = _make_internlm2_wrapper(base, apply_rotary)
    else:
        W = _make_internlm3_wrapper(base, apply_rotary)
    _WRAPPER_CACHE[key] = W
    _INTERNLM_WRAPPERS.append(W)
    return W


def install_internlm_decoder_fused_attention(
    model: nn.Module,
    quantizer: TurboQuantProd,
    *,
    architecture: str,
    allow_attention_subclass: bool = False,
) -> None:
    arch = architecture.strip().lower()
    if arch not in ("internlm2", "internlm3"):
        raise ValueError("internal: architecture must be internlm2 or internlm3")

    Base, modeling_mod = _resolve_base_and_module(model, arch)
    Wrap = _get_or_create_wrapper(Base, arch, modeling_mod)
    reg: Dict[Type[nn.Module], Type[nn.Module]] = {Base: Wrap}
    inner = _inner_decoder_stack(model)

    for layer in inner.layers:
        cur = layer.self_attn
        if any(isinstance(cur, w) for w in _INTERNLM_WRAPPERS):
            cur.bind_turboquant(quantizer)
            continue

        cur_type = type(cur)
        if allow_attention_subclass:
            wrap_key = _resolve_registered_attention_base(cur_type, reg)
            if wrap_key is None:
                raise TypeError(
                    f"Unsupported InternLM attention {cur_type.__name__}: expected a subclass of {Base.__name__}."
                )
        else:
            if cur_type is not Base:
                raise TypeError(
                    f"architecture={architecture!r} requires exact type {Base.__name__}, got {cur_type.__name__} "
                    f"(try allow_attention_subclass=True for FlashAttention2 / SDPA variants)."
                )
            wrap_key = Base

        assert wrap_key is not None
        W = reg[wrap_key]
        new = W(cur.config, layer_idx=cur.layer_idx)
        new.load_state_dict(cur.state_dict(), strict=True)
        dev = next(cur.parameters()).device
        dt = next(cur.parameters()).dtype
        new.to(device=dev, dtype=dt)
        new.bind_turboquant(quantizer)
        layer.self_attn = new


def uninstall_internlm_decoder_fused_attention(model: nn.Module) -> None:
    inner = _inner_decoder_stack(model)
    for layer in inner.layers:
        cur = layer.self_attn
        for w in _INTERNLM_WRAPPERS:
            if isinstance(cur, w):
                Base = w.__bases__[0]
                restored = Base(cur.config, layer_idx=cur.layer_idx)
                restored.load_state_dict(cur.state_dict(), strict=True)
                dev = next(cur.parameters()).device
                dt = next(cur.parameters()).dtype
                restored.to(device=dev, dtype=dt)
                layer.self_attn = restored
                break


def is_internlm_wrapper_module(attn: nn.Module) -> bool:
    return any(isinstance(attn, w) for w in _INTERNLM_WRAPPERS)
