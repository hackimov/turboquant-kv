"""
Smoke tests for Apple Metal backend (MPS).

These tests are skipped gracefully on non-macOS hosts or when MPS is unavailable.
"""

from __future__ import annotations

import platform
import unittest

import torch

from turboquant import TurboQuantProd


def _mps_available() -> bool:
    return platform.system() == "Darwin" and torch.backends.mps.is_available()


@unittest.skipUnless(_mps_available(), "requires macOS with torch MPS backend")
class TestMPSBackendSmoke(unittest.TestCase):
    def test_quantized_attention_fused_auto_mps(self):
        device = "mps"
        qtz = TurboQuantProd(bits=3, head_dim=32, device=device, dtype=torch.float32, seed=123)

        b, h_q, h_kv, m, n, d = 1, 4, 2, 3, 5, 32
        q = torch.randn(b, h_q, m, d, device=device, dtype=torch.float32)
        k = torch.randn(b, h_kv, n, d, device=device, dtype=torch.float32)
        v = torch.randn(b, h_kv, n, d, device=device, dtype=torch.float32)
        kv = qtz.quantize_kv(k, v, return_compressed=True)

        out = qtz.quantized_attention_fused_auto(q, kv, num_kv_heads=h_kv, causal=False, attention_mask=None)
        self.assertEqual(tuple(out.shape), (b, h_q, m, d))
        self.assertEqual(out.device.type, "mps")
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
