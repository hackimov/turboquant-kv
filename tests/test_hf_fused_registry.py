"""Registry / aliases for decoder fused attention (no CUDA required)."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "requires transformers")
class TestHFFusedRegistry(unittest.TestCase):
    def test_phi4_alias_registered(self):
        from turboquant.hf_fused_attention import (
            TurboQuantDeepseekV2Attention,
            TurboQuantDeepseekV3Attention,
            TurboQuantPhi3Attention,
            TurboQuantPhi4Attention,
            TurboQuantPhi4MultimodalAttention,
            supported_fused_attention_architectures,
        )

        names = supported_fused_attention_architectures()
        self.assertIn("phi3", names)
        self.assertIn("phi4", names)
        self.assertIn("phi4_multimodal", names)
        self.assertIn("internlm2", names)
        self.assertIn("internlm3", names)
        self.assertIn("deepseek_v2", names)
        self.assertIn("deepseek_v3", names)
        self.assertIn("deepseek", names)
        self.assertIn("deepseek_r1", names)
        self.assertIn("deepseek_r2", names)
        # Same wrapper class: Hub Phi-4 / Phi-4-mini use Phi3Attention.
        self.assertIs(TurboQuantPhi4Attention, TurboQuantPhi3Attention)
        self.assertIsNotNone(TurboQuantPhi4MultimodalAttention)
        self.assertIsNotNone(TurboQuantDeepseekV2Attention)
        self.assertIsNotNone(TurboQuantDeepseekV3Attention)


if __name__ == "__main__":
    unittest.main(verbosity=2)
