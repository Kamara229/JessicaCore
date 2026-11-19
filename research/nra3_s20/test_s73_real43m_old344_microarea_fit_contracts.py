"""S73 strict synthetic loss/head/OLD-TRAIN grouping contracts; no held data."""
import unittest

import torch

from research.nra3_s20.s72_nra2_situation_microareas import SituationMicroAreas
from research.nra3_s20.s73_fit_real43m_old344_microareas import (
    TRAIN,OLD_DIAG,GROUPS,SEED,EPOCHS,
    confusion,loss_on_group,sha,
)


def synthetic_row(n=5,k=3,w=7):
    torch.manual_seed(31)
    sources=torch.tensor([[0,2,4]]*n,dtype=torch.long)
    valid=torch.ones(n,k,w,dtype=torch.bool)
    old_joint=torch.randn(n,k,w+1)
    old_polar=torch.randn(n,k,w)
    old_cls=torch.randn(n,k,3)
    rec=torch.tensor([w,3,1,w,2][:n],dtype=torch.long)
    rej=torch.tensor([False,True,False,False,True][:n],dtype=torch.bool)
    return {
        "group":"S53",
        "n":n,
        "features":torch.randn(n,k,36),
        "word320":torch.randn(n,w,320).half(),
        "source_indices":sources,
        "target_safe":valid,
        "event_allowed":torch.ones(n,k,dtype=torch.bool),
        "old_joint":old_joint,
        "old_polar":old_polar,
        "old_cls":old_cls,
        "gold_slot":torch.tensor([1,0,1,0,2][:n],dtype=torch.long),
        "gold_recipient":rec,
        "gold_rejected":rej,
        "safe_rows":torch.ones(n,dtype=torch.bool),
    }


class TestS73OriginalOld344ShadowFit(unittest.TestCase):
    def test_precommitted_oldcohort_no_HELD32_or_EVAL(self):
        self.assertEqual(len(GROUPS),12)
        self.assertEqual(len(TRAIN),9)
        self.assertEqual(len(OLD_DIAG),3)
        self.assertEqual(tuple(GROUPS[:9]),TRAIN)
        self.assertEqual(tuple(GROUPS[9:]),OLD_DIAG)
        self.assertTrue(set(TRAIN).isdisjoint(OLD_DIAG))
        self.assertLessEqual(EPOCHS,60)
        self.assertEqual(SEED,731119)

    def test_frozen_parent_logits_still_exact_before_training(self):
        net=SituationMicroAreas()
        r=synthetic_row()
        y=net(r["features"],r["word320"],r["source_indices"],
              r["target_safe"],r["event_allowed"],
              r["old_joint"],r["old_polar"],r["old_cls"])
        self.assertTrue(torch.equal(y.target_or_none,r["old_joint"]))
        self.assertTrue(torch.equal(y.rejected_edge,r["old_polar"]))
        self.assertTrue(torch.equal(y.event_factuality,r["old_cls"]))
        self.assertEqual(y.output_gate,0)

    def test_training_loss_allows_null_and_rejected_without_parent_gradient(self):
        net=SituationMicroAreas()
        r=synthetic_row()
        val,n=loss_on_group(net,r)
        self.assertEqual(n,5)
        self.assertTrue(torch.isfinite(val))
        val.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in net.parameters()))
        for x in (r["old_joint"],r["word320"],r["features"]):
            self.assertIsNone(x.grad)

    def test_rejection_of_no_safe_old_native_source(self):
        net=SituationMicroAreas()
        r=synthetic_row()
        r["safe_rows"].zero_()
        with self.assertRaisesRegex(ValueError,"no original BPE-safe"):
            loss_on_group(net,r)

    def test_old_diagnostic_metrics_stable_no_grad(self):
        net=SituationMicroAreas()
        r=synthetic_row()
        base=confusion([r])
        shadow=confusion([r],net)
        self.assertEqual(base,shadow)
        self.assertEqual(base["safe"],5)
        self.assertEqual(base["gold_edges"],3)
        self.assertEqual(base["nontransfer_gold"],2)
        self.assertTrue(0<=base["recipient"]["f1"]<=1)
        self.assertTrue(0<=base["rejected"]["f1"]<=1)

if __name__=="__main__":
    unittest.main()
