# Exact upstream edits (vLLM)

**Preferred:** apply [`patches/vllm_turboquant_e38817f.patch`](patches/vllm_turboquant_e38817f.patch) with [`apply_to_vllm.py`](apply_to_vllm.py). The sections below are the same changes as **search anchors** if you port to another branch by hand.

Patch base commit: **`e38817f`** (vLLM `main` at patch time); файл патча: [`patches/vllm_turboquant_e38817f.patch`](patches/vllm_turboquant_e38817f.patch).

---

## 1. `vllm/config/cache.py`

**`CacheDType`**: add `"turboquant"` to the `Literal[...]` union.

**`CacheConfig`**: add (after the `cache_dtype` field doc block):

```python
turboquant_bits: int = Field(default=3, ge=2, le=4)
"""Bit width for TurboQuant KV when ``cache_dtype == \"turboquant\"`` (paper-style 2–4)."""
```

**`@field_validator("cache_dtype", ...)`** (existing `_validate_cache_dtype`): if `cache_dtype == "turboquant"`, log once that TurboQuant needs `pip install 'turboquant-kv[triton]'`.

---

## 2. `vllm/utils/torch_utils.py`

In `STR_DTYPE_TO_TORCH_DTYPE`, add:

```python
"turboquant": torch.uint8,
```

In `kv_cache_dtype_str_to_dtype`, before the final `STR_DTYPE_TO_TORCH_DTYPE[...]` return:

```python
if kv_cache_dtype == "turboquant":
    return torch.uint8
```

---

## 3. `vllm/v1/attention/backends/registry.py`

In `AttentionBackendEnum`, add:

```python
TURBOQUANT_ATTN = (
    "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"
)
```

---

## 4. `vllm/platforms/cuda.py`

At the very beginning of `_get_backend_priorities` (right after the docstring, **before** `if use_mla:`):

```python
if kv_cache_dtype == "turboquant":
    if use_mla:
        return []
    return [AttentionBackendEnum.TURBOQUANT_ATTN]
```

---

## 5. `vllm/v1/kv_cache_interface.py`

After the `FullAttentionSpec` class (before `MLAAttentionSpec`), insert **`TurboQuantAttentionSpec`** — same implementation as in the patch (subclass of `FullAttentionSpec`, `real_page_size_bytes` via `turboquant.vllm_pack.turboquant_paged_block_bytes`, custom `merge`).

`Self` is already imported from `typing_extensions` in current vLLM.

---

## 6. `vllm/model_executor/layers/attention/attention.py`

**Imports**: add `TurboQuantAttentionSpec` next to `FullAttentionSpec`.

**After** `self.has_sink = extra_impl_args.get("sinks") is not None`, reject `turboquant` with sliding window or `head_size_v != head_size`.

**`get_kv_cache_spec`**: in the non-sliding branch, if `vllm_config.cache_config.cache_dtype == "turboquant"`, return `TurboQuantAttentionSpec(..., dtype=torch.uint8, sliding_window=None, attention_chunk_size=None, aux_storage_dtype=vllm_config.model_config.dtype)`; else keep `FullAttentionSpec(...)`.

---

## 7. `vllm/v1/worker/utils.py`

Import `TurboQuantAttentionSpec`. In `BlockIds::init_meta`, extend the spec guard so TurboQuant pages participate in zero-block metadata (same as exact `FullAttentionSpec` today):

```python
if type(spec) is not FullAttentionSpec and type(spec) is not TurboQuantAttentionSpec:
    continue
```

---

## 8. `vllm/engine/arg_utils.py`

- **`EngineArgs`**: `turboquant_bits: int = get_field(CacheConfig, "turboquant_bits")`
- **CLI**: `cache_group.add_argument("--turboquant-bits", **cache_kwargs["turboquant_bits"])`
- **`CacheConfig(...)`** in `create_engine_config`: pass `turboquant_bits=self.turboquant_bits`

---

## 9. New file

Add `vllm/v1/attention/backends/turboquant_attn.py` — copy from [`overlay/vllm/v1/attention/backends/turboquant_attn.py`](overlay/vllm/v1/attention/backends/turboquant_attn.py).

---

## 10. Optional: docs / packaging

- Document optional dependency on `turboquant-kv[triton]`.
- Note unsupported combinations (MLA, sinks, KV sharing, non-CUDA platforms).

---

## Verification

```bash
pip install "turboquant-kv[triton]" -e ./vllm
python -c "from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend; print(TurboQuantAttentionBackend.get_name())"
```

Then `vllm serve <small_llama_like_model> --kv-cache-dtype turboquant --max-model-len 4096`.
