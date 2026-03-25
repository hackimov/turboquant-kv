"""
``model.generate`` with ``TurboQuantDynamicCache`` (compressed KV as HF ``Cache``).

Requires: ``pip install turboquant-kv[hf]`` (and with ``--fused`` on GPU — ``pip install turboquant-kv[triton]``).

By default a tiny test model is downloaded from the Hub (~a few MB). Offline / no network:

    python examples/hf_generate_turboquant_cache.py --from-config
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="HF model id (ignored with --from-config)",
    )
    p.add_argument("--from-config", action="store_true", help="Random tiny Llama from config; no Hub weights.")
    p.add_argument("--prompt", default="Hello", help="Prompt when using a real tokenizer")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--bits", type=int, default=3, choices=(2, 3, 4))
    p.add_argument(
        "--strict-reencode",
        action="store_true",
        help="Re-quantize full KV each step (slower, closer to one-shot quantize_kv)",
    )
    p.add_argument(
        "--hybrid-float-cache",
        action="store_true",
        help="TurboQuantCacheLayer: keep float KV + skip full decompress each step when strict_reencode off (see README)",
    )
    p.add_argument(
        "--fused",
        action="store_true",
        help="Triton fused attention on compressed KV (CUDA + Triton; see hf_fused_attention)",
    )
    p.add_argument(
        "--fused-arch",
        default="auto",
        metavar="NAME",
        help='HF stack name for install (e.g. llama, mistral, qwen2) or "auto" from the first layer',
    )
    p.add_argument(
        "--allow-attention-subclass",
        action="store_true",
        help="install via MRO: subclass of a registered attention (see allow_attention_subclass in hf_fused_attention)",
    )
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


def main() -> int:
    import torch

    from turboquant import TurboQuantModel

    args = _parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    if args.from_config:
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=320,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            rms_norm_eps=1e-5,
        )
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = "eager"
        model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype).eval()
        tokenizer = None
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Install transformers: pip install turboquant-kv[hf]", file=sys.stderr)
            return 1
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
        ).to(device=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    cfg = model.config
    head_dim = int(
        getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    )
    wrapper = TurboQuantModel(model, bits=args.bits, head_dim=head_dim, device=device, dtype=dtype)

    use_fused = bool(args.fused and device == "cuda")
    if args.fused and device != "cuda":
        print("Note: --fused ignored (need CUDA).", file=sys.stderr)

    cache = wrapper.make_dynamic_cache(
        triton_fused_layers=use_fused,
        strict_reencode=bool(args.strict_reencode),
        hybrid_float_cache=bool(args.hybrid_float_cache),
    )
    if use_fused:
        wrapper.enable_decoder_fused_attention(
            architecture=str(args.fused_arch),
            allow_attention_subclass=bool(args.allow_attention_subclass),
        )

    if tokenizer is None:
        input_ids = torch.randint(0, model.config.vocab_size, (1, 6), device=device)
    else:
        enc = tokenizer(args.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    pad_id = getattr(model.config, "eos_token_id", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.max_new_tokens),
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
            pad_token_id=pad_id,
        )

    print("output shape:", tuple(out.shape))
    if tokenizer is not None:
        print(tokenizer.decode(out[0], skip_special_tokens=True))
    if use_fused:
        wrapper.disable_decoder_fused_attention()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
