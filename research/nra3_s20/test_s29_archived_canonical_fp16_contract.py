"""S29 source-only canonical numeric identity tests (not original43M evidence)."""
from __future__ import annotations
import inspect
import unittest

import torch

from research.nra3_s19.paid_grammar_parser import CompactWordGrammar
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import PaidSemanticEventGraph
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import (
    canonical_paid_semantic_fp16_sha,
)
from research.nra3_s20.s29_archived_original43m_canonical_fp16_identity import (
    FROZEN_ARCHIVES,
)


class S29CanonicalFP16OnlyContracts(unittest.TestCase):
    def _fake_model(self):
        return PaidSemanticEventGraph(PaidContextualEventScope(
            CompactWordGrammar().eval().requires_grad_(False)
        ).eval().requires_grad_(False)).eval()

    def test_canonical_FP16_fingerprint_stable_across_repeated_legacy_serialization(self):
        m=self._fake_model()
        old=canonical_paid_semantic_fp16_sha(m)
        self.assertEqual(len(old),64)
        for _ in range(5):
            archive=m.physical_bytes()
            restored=PaidSemanticEventGraph.from_physical(archive)
            self.assertEqual(canonical_paid_semantic_fp16_sha(restored),old)
        self.assertEqual(m.output_gate,0)

    def test_canonical_fingerprint_detects_paid_weight_change(self):
        m=self._fake_model()
        old=canonical_paid_semantic_fp16_sha(m)
        with torch.no_grad():
            m.role.weight[0,0].add_(1.)
        new=canonical_paid_semantic_fp16_sha(m)
        self.assertNotEqual(new,old)

    def test_original_model_SHA_pins_and_no_forward_train_eval_paths_in_archive_audit(self):
        self.assertEqual(FROZEN_ARCHIVES["S22d"]["full_paid_params"],12381)
        self.assertEqual(FROZEN_ARCHIVES["S26"]["full_paid_params"],12384)
        import research.nra3_s20.s29_archived_original43m_canonical_fp16_identity as module
        src=inspect.getsource(module.audit_actual_archived_model)
        self.assertNotIn("optimizer(",src)
        self.assertNotIn("backward(",src)
        self.assertNotIn("model.forward(",src)
        self.assertNotIn("EVAL128",src)
        self.assertIn("output_gate==0",src)
        self.assertIn("canonical_paid_semantic_fp16_sha",src)


if __name__=="__main__":
    unittest.main()
