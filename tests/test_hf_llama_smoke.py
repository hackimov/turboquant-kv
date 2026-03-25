"""
Smoke tests: tiny random Llama (config-only, no Hub weights) + TurboQuant ``DynamicCache``.

Requires ``pip install -e ".[hf]"`` (or ``transformers``). CUDA + Triton tests are skipped on CPU.
"""

from __future__ import annotations

import importlib.util
import unittest

import torch

from turboquant import TurboQuantProd


def _have_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _triton_cuda() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _tiny_llama_config():
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
    )
    # Portable across transformers 4.48+: constructor may not accept this kwarg on all versions.
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    return cfg


def _tiny_llama_model(device: str):
    from transformers import LlamaForCausalLM

    cfg = _tiny_llama_config()
    model = LlamaForCausalLM(cfg)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


@unittest.skipUnless(_have_transformers(), "requires transformers (pip install turboquant-kv[hf])")
class TestHFLlamaTurboQuantSmoke(unittest.TestCase):
    def test_prefill_decode_dynamic_cache_cpu(self):
        device = "cpu"
        model = _tiny_llama_model(device)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=head_dim, device=device, dtype=torch.float32, seed=42)

        from turboquant.hf_cache import turboquant_dynamic_cache

        cache = turboquant_dynamic_cache(model.config, q, strict_reencode=False)
        inp = torch.randint(0, model.config.vocab_size, (1, 6), device=device)

        with torch.no_grad():
            model(input_ids=inp[:, :4], past_key_values=cache, use_cache=True)
        self.assertEqual(cache.get_seq_length(0), 4)

        with torch.no_grad():
            out = model(input_ids=inp[:, 4:5], past_key_values=cache, use_cache=True)
        self.assertEqual(cache.get_seq_length(0), 5)
        self.assertTrue(torch.isfinite(out.logits).all())

    def test_install_uninstall_restores_llama_attention_type(self):
        from transformers.models.llama.modeling_llama import LlamaAttention

        from turboquant.hf_llama_fused import (
            TurboQuantLlamaAttention,
            install_turboquant_llama_attention,
            uninstall_turboquant_llama_attention,
        )

        model = _tiny_llama_model("cpu")
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=head_dim, device="cpu", dtype=torch.float32, seed=5)
        install_turboquant_llama_attention(model, q)
        for lay in model.model.layers:
            self.assertIsInstance(lay.self_attn, TurboQuantLlamaAttention)
        uninstall_turboquant_llama_attention(model)
        for lay in model.model.layers:
            self.assertIs(type(lay.self_attn), LlamaAttention)

    def test_triton_fused_cache_cpu_falls_back_to_eager_attn(self):
        """TurboQuantLlamaAttention on CPU: fused path off, HF still runs (super().forward + cache.update)."""
        device = "cpu"
        model = _tiny_llama_model(device)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=head_dim, device=device, dtype=torch.float32, seed=43)

        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_llama_fused import (
            install_turboquant_llama_attention,
            uninstall_turboquant_llama_attention,
        )

        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_llama_attention(model, q)
        try:
            inp = torch.randint(0, model.config.vocab_size, (1, 5), device=device)
            with torch.no_grad():
                model(input_ids=inp[:, :3], past_key_values=cache, use_cache=True)
                out = model(input_ids=inp[:, 3:4], past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
        finally:
            uninstall_turboquant_llama_attention(model)

    @unittest.skipUnless(_triton_cuda(), "requires CUDA and Triton")
    def test_prefill_decode_fused_path_cuda(self):
        device = "cuda"
        model = _tiny_llama_model(device)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=head_dim, device=device, dtype=torch.float32, seed=44)

        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_llama_fused import (
            install_turboquant_llama_attention,
            uninstall_turboquant_llama_attention,
        )

        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_llama_attention(model, q)
        try:
            inp = torch.randint(0, model.config.vocab_size, (1, 7), device=device)
            with torch.no_grad():
                model(input_ids=inp[:, :5], past_key_values=cache, use_cache=True)
                out = model(input_ids=inp[:, 5:6], past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
        finally:
            uninstall_turboquant_llama_attention(model)

    @unittest.skipUnless(_triton_cuda(), "requires CUDA and Triton")
    def test_fused_path_with_padding_batch_cuda(self):
        """HF causal+padding mask must match fused [B,H,M,N]; forward completes without NaNs."""
        device = "cuda"
        model = _tiny_llama_model(device)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=head_dim, device=device, dtype=torch.float32, seed=7)

        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_llama_fused import (
            install_turboquant_llama_attention,
            uninstall_turboquant_llama_attention,
        )

        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_llama_attention(model, q)
        try:
            input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], device=device)
            attn = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
            next_tok = torch.tensor([[9], [8]], device=device)
            attn2 = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.long, device=device)
            pos = torch.tensor([[3], [3]], device=device)
            with torch.no_grad():
                out2 = model(
                    input_ids=next_tok,
                    attention_mask=attn2,
                    past_key_values=out.past_key_values,
                    use_cache=True,
                    position_ids=pos,
                )
            self.assertTrue(torch.isfinite(out2.logits).all())
        finally:
            uninstall_turboquant_llama_attention(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
