"""S85 new TRAIN72 grouped 56/16 source-only and frozen-parent ABI tests.

Synthetic 320D tensors only, NOT physical model performance.
"""
import ast
import unittest
from pathlib import Path

import torch

from research.nra3_s20.s82_predicate_centered_situation_fields import (
    S82PredicateCenteredMiniAreas,
)
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,TRAIN_GROUPS,DIAG_GROUPS,
)
from research.nra3_s20.s85_true43mGT_S82_event_scope_train_new72 import (
    N,K,PAID,SEED,EPOCHS,S83_SHA,STAGE,
)


class TestS85NewTrainOnlyOriginalSourceWall(unittest.TestCase):
    def test_locked_S83_newTRAIN72_fixed_grouping(self):
        self.assertEqual(N,72)
        self.assertEqual(K,8)
        self.assertEqual(S83_SHA,source_only_new72_sha256())
        rows=source_only_new72()
        self.assertEqual(len([r for r in rows if r["group"] in TRAIN_GROUPS]),56)
        self.assertEqual(len([r for r in rows if r["group"] in DIAG_GROUPS]),16)
        self.assertTrue(set(TRAIN_GROUPS).isdisjoint(DIAG_GROUPS))
        self.assertEqual(SEED,851119)
        self.assertEqual(EPOCHS,90)
        self.assertIn("S82_EVENT_SCOPE_8264",STAGE)

    def test_only_eight_new_models_and_zero_activation(self):
        m=S82PredicateCenteredMiniAreas()
        self.assertEqual(m.budget()["new_neural_parameters"],PAID)
        self.assertEqual(PAID,8264)
        self.assertEqual(len(m.areas),8)
        self.assertFalse(m.is_trained)
        self.assertFalse(m.activation_allowed)
        self.assertFalse(m.candidate_accepted)
        self.assertEqual(m.output_gate,0)

    def test_gold_module_import_only_in_separate_loss_path(self):
        tree=ast.parse(Path(
            "research/nra3_s20/s85_true43mGT_S82_event_scope_train_new72.py"
            ).read_text(encoding="utf8"))
        top_imports=[z.module or "" for z in tree.body if isinstance(z,ast.ImportFrom)]
        self.assertTrue(all("s83_NEW_DIVERSE72_gold" not in z
                            and "s79_NEW_NOVEL48" not in z
                            and "s81_frozen" not in z
                            for z in top_imports))
        infer=next(z for z in tree.body
                   if isinstance(z,ast.FunctionDef)
                   and z.name=="fully_frozen_original_SOURCE_only")
        self.assertFalse(any(isinstance(z,ast.ImportFrom) and
            "gold" in (z.module or "").lower() for z in ast.walk(infer)))
        unseal=next(z for z in tree.body
                    if isinstance(z,ast.FunctionDef)
                    and z.name=="open_constructed_targets_ONLY_after_source_forward")
        self.assertTrue(any(isinstance(z,ast.ImportFrom)
            and "s83_NEW_DIVERSE72_gold_TRAIN_LOSS_ONLY" in (z.module or "")
            for z in ast.walk(unseal)))

    def test_not_reuse_spent_S79_as_training_feature_or_optimizer_target(self):
        p=Path("research/nra3_s20/s85_true43mGT_S82_event_scope_train_new72.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        inputs=[z.module or "" for z in ast.walk(tree)
                if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s79_NEW_NOVEL48" in z
                            or "s81_frozen_S78" in z
                            or "s66_held_gold" in z
                            for z in inputs))

    def test_no_true43M_gradients_when_s82_train_head_receives_detached_features(self):
        text=source_only_new72()[0]["text"]
        from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
        tokens=raw_surface_grammar(text); w=len(tokens)
        m=S82PredicateCenteredMiniAreas()
        hidden=torch.randn(1,w,320)
        valid=torch.ones(1,w,dtype=torch.bool)
        unsafe=torch.zeros_like(valid)
        idx=torch.tensor([[3]],dtype=torch.long)
        okay=torch.ones(1,1,dtype=torch.bool)
        y=m(hidden,valid,unsafe,idx,okay,(text,))
        self.assertEqual(y.shape,(1,1,8))
        y.pow(2).mean().backward()
        self.assertIsNone(hidden.grad)
        self.assertTrue(all(p.grad is not None for p in m.parameters()))


if __name__=="__main__":
    unittest.main()
