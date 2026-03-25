# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# TurboQuant compressed KV + Triton fused decode (optional dependency: ``turboquant-kv[triton]``).
# Copy this file into upstream vLLM at the same path, then apply ``UPSTREAM_EDITS.md``.

from __future__ import annotations

from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
logger = init_logger(__name__)


class TurboQuantAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    """Reuse Triton workspace buffers and metadata layout."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER


class TurboQuantAttentionBackend(AttentionBackend):
    """Paged TurboQuant storage; decode via ``turboquant`` Triton fused attention."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["turboquant"]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TurboQuantAttentionMetadataBuilder]:
        return TurboQuantAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str != "turboquant":
            raise ValueError("TurboQuant backend expects cache_dtype_str='turboquant'")
        from turboquant.vllm_pack import turboquant_paged_block_bytes

        page_b = turboquant_paged_block_bytes(
            block_size, num_kv_heads, head_size, torch.float16
        )
        return (num_blocks, page_b)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [16, 32, 64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(7, 5)

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: DeviceCapability,
        attn_type: str,
    ) -> list[str]:
        invalid = super().validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            use_per_head_quant_scales=use_per_head_quant_scales,
            device_capability=device_capability,
            attn_type=attn_type,
        )
        if kv_cache_dtype == "turboquant":
            try:
                import triton  # noqa: F401
                import turboquant  # noqa: F401
            except ImportError as e:
                invalid.append(f"turboquant with triton required: {e}")
        return invalid


class TurboQuantAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("TurboQuant backend supports decoder attention only")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi is not supported for TurboQuant KV")
        if sliding_window is not None:
            raise NotImplementedError("Sliding-window + TurboQuant is not implemented")
        if sinks is not None:
            raise NotImplementedError("Attention sinks are not supported for TurboQuant KV")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported for TurboQuant KV")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self._quantizer = None
        self.supports_quant_query_input = False

    def _get_quantizer(self, device: torch.device):
        dev_s = str(device)
        if self._quantizer is None or str(self._quantizer.device) != dev_s:
            from turboquant import TurboQuantProd

            cfg = get_current_vllm_config()
            bits = int(getattr(cfg.cache_config, "turboquant_bits", 3))
            self._quantizer = TurboQuantProd(
                bits=bits,
                head_dim=self.head_size,
                device=dev_s,
                dtype=torch.float32,
            )
        return self._quantizer

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        return

    def forward(
        self,
        _layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Output quantization is not supported")

        if attn_metadata is None:
            return output.fill_(0)

        assert not attn_metadata.use_cascade

        from turboquant.kernels.fused_attention import turboquant_fused_attention_paged
        from turboquant.vllm_pack import (
            TurboQuantPageLayout,
            paged_kv_views_from_allocator_buffer,
        )

        vllm_config = get_current_vllm_config()
        block_size = vllm_config.cache_config.block_size
        layout = TurboQuantPageLayout.build(
            block_size,
            self.num_kv_heads,
            self.head_size,
            query.dtype,
        )
        paged = paged_kv_views_from_allocator_buffer(kv_cache, layout)

        qz = self._get_quantizer(query.device)
        pi = qz.Pi.to(query.device, dtype=torch.float32)
        s_mat = qz.S.to(query.device, dtype=torch.float32)
        centroids = qz._centroids.to(query.device, dtype=torch.float32)

        num_reqs = attn_metadata.seq_lens.shape[0]
        qs = attn_metadata.query_start_loc
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        num_actual = attn_metadata.num_actual_tokens
        max_ctx = attn_metadata.max_seq_len

        for b in range(num_reqs):
            s = int(qs[b].item())
            e = int(qs[b + 1].item())
            if e <= s:
                continue
            q_seg = query[s:e].permute(1, 0, 2).unsqueeze(0).contiguous().to(torch.float32)
            q_pi = torch.matmul(q_seg, pi.T)
            q_s = torch.matmul(q_seg, s_mat.T)
            bt = block_table[b : b + 1]
            sl = seq_lens[b : b + 1]
            out = turboquant_fused_attention_paged(
                q_pi,
                q_s,
                paged,
                bt,
                sl,
                block_size,
                max_ctx,
                centroids=centroids,
                qjl_factor=float(qz._qjl_factor),
                pi=pi,
                s=s_mat,
                num_kv_heads=self.num_kv_heads,
                causal=True,
                attention_mask=None,
            )
            out = out.squeeze(0).permute(1, 0, 2).to(dtype=output.dtype)
            output[s:e].copy_(out.reshape(e - s, -1))

        if num_actual < output.shape[0]:
            output[num_actual:].zero_()
        return output

    def do_kv_cache_update(
        self,
        _layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        from turboquant.vllm_pack import TurboQuantPageLayout, scatter_tokens_from_cache_update

        cfg = get_current_vllm_config()
        block_size = cfg.cache_config.block_size
        layout = TurboQuantPageLayout.build(
            block_size,
            self.num_kv_heads,
            self.head_size,
            key.dtype,
        )
        qz = self._get_quantizer(key.device)
        scatter_tokens_from_cache_update(
            kv_cache.view(torch.uint8),
            layout,
            key,
            value,
            slot_mapping,
            qz,
            block_size,
        )
