"""S88 pretrained S85 firstread protocol, pure synthetic ABI unit tests only."""
import ast
import unittest
from pathlib import Path
import torch

from research.nra3_s20.s82_predicate_centered_situation_fields import (
    S82PredicateCenteredMiniAreas,
)
from research.nra3_s20.s86_newSYNTHETIC48_SOURCEONLY_never_train import (
    S86_novel48_source_only,S86_novel48_source_sha256,
)
from research.nra3_s20.s88_S85_FROZEN_REAL43M_S86_new48_FIRSTREAD import (
    SOURCE_SHA,S85_PAID_SHA,S87_PHYSICAL_SHA,STAGE,N,K,PAID,micro,
    FIRST_source_only_frozen_S85_logits_BEFORE_any_target_read,
    posthoc_ONE_SHOT_gold_after_logits_saved,
)


class TestS88FirstReadModelGoldIsolation(unittest.TestCase):
    def test_preregistered_source_and_S85_weights_pinned(self):
        self.assertEqual(SOURCE_SHA,S86_novel48_source_sha256())
        self.assertEqual(len(S86_novel48_source_only()),48)
        self.assertEqual(S85_PAID_SHA,
            "45997dca99a91c820e285536d5b1ce90c3567db1a58e244c6d26072380e06eaf")
        self.assertEqual((N,K,PAID),(48,8,8264))
        self.assertIn("ONE_SHOT_FIRSTREAD",STAGE)

    def test_original_8_neural_heads_accept_48_not_force_teacher_gold(self):
        torch.manual_seed(881119)
        m=S82PredicateCenteredMiniAreas().eval().requires_grad_(False)
        self.assertEqual(m.budget()["new_neural_parameters"],8264)
        rows=S86_novel48_source_only()
        texts=tuple(x["text"] for x in rows)
        from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
        width=max(len(raw_surface_grammar(x)) for x in texts)
        x=torch.randn(N,width,320)
        valid=torch.zeros((N,width),dtype=torch.bool)
        for i,t in enumerate(texts):
            valid[i,:len(raw_surface_grammar(t))]=True
        unsafe=torch.zeros_like(valid)
        # Source-only proposed event indices are intentionally NOT correct
        # answer anchors: this is only a functional batch/ABI smoke test.
        idx=torch.zeros((N,K),dtype=torch.long)
        allow=torch.ones((N,K),dtype=torch.bool)
        with torch.inference_mode():
            out=m(x,valid,unsafe,idx,allow,texts)
        self.assertEqual(out.shape,(N,K,8))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.activation_allowed)

    def test_source_only_before_gold_module_import(self):
        file=Path("research/nra3_s20/s88_S85_FROZEN_REAL43M_S86_new48_FIRSTREAD.py")
        tree=ast.parse(file.read_text(encoding="utf8"))
        toplevel=[x.module or "" for x in tree.body
                  if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s86_NOVEL48_GOLD" in x or
                             "s79_NEW_NOVEL48" in x or
                             "s81_frozen" in x for x in toplevel))
        infer=next(x for x in tree.body
                   if isinstance(x,ast.FunctionDef)
                   and x.name=="FIRST_source_only_frozen_S85_logits_BEFORE_any_target_read")
        self.assertFalse(any(isinstance(x,ast.ImportFrom) and
                             "GOLD" in (x.module or "")
                             for x in ast.walk(infer)))
        post=next(x for x in tree.body
                  if isinstance(x,ast.FunctionDef)
                  and x.name=="posthoc_ONE_SHOT_gold_after_logits_saved")
        self.assertTrue(any(isinstance(x,ast.ImportFrom) and
                            "s86_NOVEL48_GOLD_SEPARATE" in (x.module or "")
                            for x in ast.walk(post)))

    def test_micro_scores_fixed_threshold_no_model_refit(self):
        truth=torch.tensor([[True,False,False,True,False,False,False,False],
                            [False,True,False,False,False,False,False,True]])
        r=micro(truth.clone(),truth)
        self.assertEqual(r["micro_F1"],1.)
        self.assertEqual(r["all8_factor_exact"],2)
        worse=truth.clone();worse[0,0]=False
        rw=micro(worse,truth)
        self.assertLess(rw["micro_F1"],1.)
        self.assertEqual(rw["all8_factor_exact"],1)

    def test_no_optimizer_or_checkpoint_selection_in_firstread_path(self):
        p=Path("research/nra3_s20/s88_S85_FROZEN_REAL43M_S86_new48_FIRSTREAD.py")
        t=p.read_text(encoding="utf8")
        self.assertNotIn("AdamW(",t)
        self.assertNotIn("optimizer.step(",t)
        self.assertNotIn("loss.backward(",t)
        self.assertIn("torch.inference_mode()",t)
        self.assertIn("net.load_state_dict(",t)


if __name__=="__main__":
    unittest.main()
