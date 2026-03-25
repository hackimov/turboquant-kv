# TurboQuant → vLLM (upstream) integration

End-to-end wiring for **vLLM v1** with packed TurboQuant KV pages and Triton fused decode (`turboquant` + `turboquant.vllm_pack`). After applying the patch below you get:

- `--kv-cache-dtype turboquant` and `--turboquant-bits {2,3,4}` (default 3)
- CUDA backend selection → `TURBOQUANT_ATTN` when the KV dtype is `turboquant`
- `TurboQuantAttentionSpec` for correct `real_page_size_bytes` in the KV allocator
- Backend module `vllm/v1/attention/backends/turboquant_attn.py` (same content as `overlay/...` in this repo)

## Requirements

- GPU with Triton support (same class of devices as the rest of TurboQuant CUDA paths)
- `pip install turboquant-kv[triton]` in the **same** environment as vLLM
- Models with **decoder** attention, **no sliding window**, **no sinks**, **no KV sharing**, **no MLA**, `head_size` in `{16,32,64,128,256}`, and `head_size_v == head_size`

## Install (recommended: git patch)

1. Clone [vLLM](https://github.com/vllm-project/vllm) and check out a revision close to patch base commit **`e38817f`** (vLLM `main` when [`patches/vllm_turboquant_e38817f.patch`](patches/vllm_turboquant_e38817f.patch) was generated). From the vLLM repo root you can `git apply` that file; the script below is recommended.
2. From the **turboquant-kv** repo:

   ```bash
   python integrations/vllm_upstream/apply_to_vllm.py /path/to/vllm
   ```

   Dry-run: add `--check`. If hunks fail on a newer main, try `--3way`.

3. `pip install turboquant-kv[triton]` then install vLLM in editable mode, e.g. `pip install -e ./vllm`.
4. Smoke:

   ```bash
   vllm serve <model> --kv-cache-dtype turboquant --max-model-len 4096
   # optional: --turboquant-bits 4
   ```

## Install (manual / porting)

If the patch does not apply, use [`UPSTREAM_EDITS.md`](UPSTREAM_EDITS.md) as a checklist (same edits as the patch, with file anchors). Copy `overlay/vllm/v1/attention/backends/turboquant_attn.py` if that file is missing.

## Layout and tests (this repository)

- Page packing and scatter helpers: `turboquant.vllm_pack` — tests in `tests/test_vllm_pack.py` and CUDA paged-attention checks in `tests/test_triton_fused_attention.py`.
- vLLM stores one `uint8` row per paged block. For Triton decode, use **`paged_kv_views_from_allocator_buffer(kv_cache, layout)`** (recommended in the overlay) or **`uint8_pages_to_paged_dict`**: both build zero-copy `*_phys` views `[P, block_size, H_kv, D]` for `turboquant_fused_attention_paged`.
- The paged fused kernel tiles query rows (`BLOCK_M = 16`), masks invalid physical page ids (`pb < 0` or `pb >= P`) and out-of-range logical block columns, and avoids reading `block_tables` past the tensor width—so allocator padding (`-1` slots, wide `block_table`) matches real vLLM usage.

## Limitations

- **ROCm / CPU / XPU**: not wired; CUDA `_get_backend_priorities` only selects TurboQuant when `kv_cache_dtype == "turboquant"`.
- **Prefix caching / CUDA graphs**: not fully validated; prefer `--enforce-eager` for first bring-up if you see capture errors.
- **Performance**: per-token cache updates run in Python; a future step is a fused CUDA/Triton cache writer.

## Upstream PR checklist

1. Apache-2.0 headers on new files (already in `turboquant_attn.py`).
2. Optional dependency: document `turboquant-kv[triton]` in vLLM docs or extras.
3. CI: smoke test gated on CUDA + optional import (pattern used for other Triton backends).
4. Keep `TurboQuantAttentionSpec` and worker `ZeroBlockIds` / `init_meta` handling in sync (`type(spec) is TurboQuantAttentionSpec` branch in `vllm/v1/worker/utils.py`).
