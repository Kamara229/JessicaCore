"""S81 FIRST-READ sealed comparator contract; SYNTHETIC tensors ONLY.

No frozen actual S78 model/weights, no real novel48 H8 and no gold scores
are inspected by these unit tests. First actual result one-time in CI.
"""
import ast
import unittest
from pathlib import Path
import torch

from research.nra3_s20.s81_frozen_S78_firstread_S79_novel48_once import (
    _micro_and_exact,S79_SOURCE_SHA,S78_PAID_SHA,N,K,PHYSICAL_S78_BATCH,
    STAGE,BEFORE_GOLD,FINAL,
)
from research.nra3_s20.s78_train_true43m_quote_mention_8_specialists import (
    S78MentionSituationSpecialists,
)
from research.nra3_s20.s79_NEW_NOVEL48_SOURCE_ONLY_locked_before_S78_firstread import (
    new_novel48_source_sha256,
)


class TestS81Novel48FirstReadNoTuning(unittest.TestCase):
    def test_new_SOURCE_sha_and_existing_REAL_S78_checkpoint_sha_strict(self):
        self.assertEqual(S79_SOURCE_SHA,new_novel48_source_sha256())
        self.assertEqual(S78_PAID_SHA,
            "7e642eb3e5a9eecef75c07d7ff16bc6065760f1d1fa93f0abaebe254a5ea5749")
        self.assertEqual((N,K,PHYSICAL_S78_BATCH),(48,8,120))
        self.assertIn("FIRST_READ_FROZEN_S78",STAGE)

    def test_120_batch_padding_is_per_row_independent(self):
        torch.manual_seed(81)
        net=S78MentionSituationSpecialists().eval().requires_grad_(False)
        x=torch.randn(48,7,339)
        x[:,:,320:339]=0.
        valid=torch.ones(48,7,dtype=torch.bool)
        ranked=torch.randint(0,7,(48,8))
        okay=torch.ones(48,8,dtype=torch.bool)
        pad1=torch.arange(120)%48
        pad2=torch.cat((torch.arange(48),torch.zeros(72,dtype=torch.long)))
        with torch.inference_mode():
            a=net(x[pad1],valid[pad1],ranked[pad1],okay[pad1])[:48]
            b=net(x[pad2],valid[pad2],ranked[pad2],okay[pad2])[:48]
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(a.shape,(48,8,8))
        self.assertFalse(net.activation_allowed)

    def test_source_model_firstread_cannot_import_gold_top_level(self):
        text=Path("research/nra3_s20/s81_frozen_S78_firstread_S79_novel48_once.py").read_text()
        tree=ast.parse(text)
        imports=[z.module or "" for z in tree.body
                 if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("LABELS_POSTHOC" in name for name in imports))
        self.assertFalse(any("s79_NEW_NOVEL48_LABELS" in name for name in imports))
        self.assertIn("NRA2_S81_PRE_GOLD",BEFORE_GOLD)
        self.assertIn("POSTHOC",FINAL)
        post=next(z for z in tree.body if isinstance(z,ast.FunctionDef)
                  and z.name=="score_once_POSTHOC_only")
        self.assertTrue(any(isinstance(z,ast.ImportFrom) and
                            "s79_NEW_NOVEL48_LABELS_POSTHOC" in (z.module or "")
                            for z in ast.walk(post)))

    def test_predeclared_micro_f1_and_exact_no_oracle_selection(self):
        target=torch.tensor([[True,False,True,False,False,False,False,False],
                             [False,True,False,False,False,False,False,False]])
        pred=target.clone()
        x=_micro_and_exact(pred,target)
        self.assertEqual(x["micro_f1"],1.)
        self.assertEqual(x["exact_eight_factors_correct"],2)
        pred[0,0]=False
        x=_micro_and_exact(pred,target)
        self.assertLess(x["micro_f1"],1.)
        self.assertEqual(x["exact_eight_factors_correct"],1)


if __name__=="__main__":
    unittest.main()
