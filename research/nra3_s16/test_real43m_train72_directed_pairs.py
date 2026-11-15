"""NRA3 S16b regression: physically loaded pair score == full causal S16."""
from __future__ import annotations
import hashlib
import unittest

import torch
from research.nra3_s16.paid_directed_valence import (
    PaidDirectedValence,PARAMETERS,PHYSICAL_BYTES,ROLES
)
from research.nra3_s16.real43m_train72_directed_valence import (
    PRECOMMITTED_STEPS,CHECKPOINT_STEPS,LEARNING_RATE,
    DirectedPairTrain,metrics,score_weighted_train,
    S16_TRAIN_SOURCE_SHA
)
from research.nra3_s16.direct_train72_edges import (
    sealed_directed_train72
)


class S16PhysicalTrainingPairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(316022)
        torch.set_num_threads(1)

    def test_paid_PREDICATE_CURRENT_then_ARGUMENT_OLDER_exact_full_forward(self):
        model=PaidDirectedValence().eval()
        physical=model.physical_bytes()
        other=PaidDirectedValence.from_physical(physical)
        r=torch.randn(1,8,128,dtype=torch.float32)
        mask=torch.ones((1,8),dtype=torch.bool)
        with torch.inference_mode():
            full=other(r,mask)
            s=other.score_pairs(
                r[0,5:6],r[0,2:3],
                torch.tensor([5]),torch.tensor([2]),torch.tensor([5]))
        self.assertTrue(torch.allclose(
            s[0],full.predicate_now[0,5,2],atol=1e-5,rtol=1e-5))
        self.assertEqual(s.shape,(1,len(ROLES)))

    def test_paid_ARGUMENT_CURRENT_then_PREDICATE_OLDER_same_gold_link_direction(self):
        m=PaidDirectedValence.from_physical(
            PaidDirectedValence().physical_bytes())
        r=torch.randn(1,9,128)
        v=torch.ones(1,9,dtype=torch.bool)
        with torch.inference_mode():
            full=m(r,v)
            s=m.score_pairs(
                r[0,1:2],r[0,6:7],
                torch.tensor([1]),torch.tensor([6]),torch.tensor([6]))
        self.assertTrue(torch.allclose(
            s[0],full.argument_now[0,6,1],atol=1e-5,rtol=1e-5))

    def test_paired_forward_invariance_to_other_later_words(self):
        m=PaidDirectedValence()
        r=torch.randn(1,8,128)
        p=torch.tensor([5],dtype=torch.long)
        a=torch.tensor([2],dtype=torch.long)
        c=torch.tensor([5],dtype=torch.long)
        first=m.score_pairs(r[0,5:6],r[0,2:3],p,a,c)
        r[:,6:,:].normal_(0,10.)
        again=m.score_pairs(r[0,5:6],r[0,2:3],p,a,c)
        self.assertTrue(torch.equal(first,again))
        with self.assertRaises(ValueError):
            m.score_pairs(r[0,5:6],r[0,2:3],p,a,torch.tensor([4]))
        with self.assertRaises(ValueError):
            m.score_pairs(r[0,5:6],r[0,2:3],p,p,c)
        with self.assertRaises(ValueError):
            m.score_pairs(r[0,5:6],r[0,2:3],p.float(),a,c)
        with self.assertRaises(ValueError):
            m.score_pairs(r[0,5:6],r[0,2:3,:-1],p,a,c)

    def test_all_2309_new_fp16_fields_get_training_gradient_no_gold_to_model(self):
        model=PaidDirectedValence()
        N=24
        p=torch.randn(N,128)
        a=torch.randn(N,128)
        pp=torch.arange(N,dtype=torch.long)+10
        ap=torch.arange(N,dtype=torch.long)
        cur=pp.clone()
        target=torch.tensor([0,1,2,3,4,0]*4,dtype=torch.long)
        group=DirectedPairTrain(p,a,pp,ap,cur,target,
                                torch.zeros(N,dtype=torch.long),
                                torch.zeros(N,dtype=torch.long),1,16)
        weight=torch.tensor([.2,1.,1.,1.,1.5])
        loss=score_weighted_train(model,group,weight)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        for z in model.parameters():
            self.assertIsNotNone(z.grad)
            self.assertTrue(bool(torch.isfinite(z.grad).all()))
            self.assertTrue(bool((z.grad!=0).any()))
        self.assertEqual(PARAMETERS,2309)
        self.assertEqual(PHYSICAL_BYTES,4665)

    def test_micro_link_F1_favors_exact_directed_role_not_NONE_accuracy(self):
        N=6
        t=torch.tensor([0,1,2,3,4,0],dtype=torch.long)
        group=DirectedPairTrain(
            torch.randn(N,128),torch.randn(N,128),
            torch.arange(N)+8,torch.arange(N),torch.arange(N)+8,
            t,torch.tensor([0,0,1,1,0,1],dtype=torch.long),
            torch.zeros(N,dtype=torch.long),1,4)
        # exact true positives two, wrong role two, none correct one
        # and falsely predicted link for the other NONE.
        pred=torch.tensor([0,1,3,0,4,2])
        logits=torch.full((N,5),-3.)
        logits[torch.arange(N),pred]=5.
        result=metrics(logits,group)
        m=result["all_two_direction_wordpair_exact_role"]
        self.assertEqual(m["gold_links"],4)
        self.assertEqual(m["exact_directed_role_true_positive"],2)
        self.assertEqual(m["false_negative"],2)
        self.assertEqual(m["false_positive"],2)
        self.assertAlmostEqual(m["precision"],.5)
        self.assertAlmostEqual(m["recall"],.5)
        self.assertAlmostEqual(m["f1"],.5)
        self.assertEqual(result["per_role"]["SOURCE"]["gold"],1)

    def test_existing_72_gold_SHA_not_any_consumed_EVAL_data(self):
        gold=sealed_directed_train72()
        self.assertEqual(
          S16_TRAIN_SOURCE_SHA,
          "9843c35ee9aa75b8f5c9c963e441589dd2f0292e41ec7933971d5aa3dcd365d6")
        self.assertEqual(gold["S16_complete_predicate_to_argument_gold_source_sha256"],
                         S16_TRAIN_SOURCE_SHA)
        self.assertEqual(gold["S16_gold_directed_links"],216)
        self.assertFalse(gold["S16_used_consumed_S15c_EVAL24"])
        self.assertFalse(gold["S16_used_consumed_S12c_EVAL16"])
        self.assertEqual(PRECOMMITTED_STEPS,200)
        self.assertEqual(CHECKPOINT_STEPS,(0,25,50,75,100,125,150,175,200))
        self.assertAlmostEqual(LEARNING_RATE,.022)


if __name__=="__main__":
    unittest.main()
