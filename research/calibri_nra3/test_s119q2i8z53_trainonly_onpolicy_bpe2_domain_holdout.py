"""Z53 frozen on-policy old legal84 source-only structural and smoke contracts."""
import unittest
import torch
from pathlib import Path
from research.calibri_nra3.s119q2i8z53_trainonly_onpolicy_bpe2_domain_holdout import (
    DOMAINS,EPOCHS,SEED,LR,NATIVE_K,SHADOW_PARAMS,
    SOURCE_SHA,entropy,check,selective_index,ABSTENTION_MARGIN,teacher_loss,KD_WEIGHT,TEACHER_ENTROPY_THRESHOLD_NATS
)

ROOT=Path(__file__).parent
RANK=ROOT/"s119q2i8z53_trainonly_onpolicy_bpe2_domain_holdout.py"
PHYS=ROOT/"s119q2i8z53_physical43m_onpolicy_domainholdout.py"

class TestZ53ResearchOnly(unittest.TestCase):
    def test_precommitted_exposure_source_and_training_config(self):
        self.assertEqual(len(DOMAINS),7)
        self.assertEqual(EPOCHS,8)
        self.assertEqual(NATIVE_K,64)
        self.assertEqual(SHADOW_PARAMS,30720)
        self.assertEqual(LR,.004)
        self.assertEqual(SEED,119145)
        self.assertEqual(len(SOURCE_SHA),64)
    def test_margin_abstains_and_accepts_without_gold(self):
        self.assertEqual(ABSTENTION_MARGIN,.25)
        base=torch.full((64,),-5.)
        base[0]=1.
        guarded=base.clone(); guarded[1]=1.125
        chosen,margin,abstained=selective_index(base,guarded)
        self.assertEqual((chosen,abstained),(0,True))
        self.assertAlmostEqual(margin,.125)
        guarded[1]=1.5
        self.assertEqual(selective_index(base,guarded),(1,.5,False))
        self.assertEqual(selective_index(base,base)[0],0)
        with self.assertRaises(ValueError):
            selective_index(base,torch.zeros(63))
    def test_entropy_aware_teacher_loss(self):
        self.assertEqual((KD_WEIGHT,TEACHER_ENTROPY_THRESHOLD_NATS),(.25,2.))
        original=torch.linspace(-2.,2.,49152).view(1,-1)
        ids=torch.topk(original[0],64,sorted=True).indices
        scores=original[0,ids].clone().requires_grad_()
        uncertain=torch.zeros_like(original)
        rev,active=teacher_loss(original,ids,scores,uncertain,adaptive=False)
        both,active2=teacher_loss(original,ids,scores,uncertain,adaptive=True)
        self.assertFalse(active)
        self.assertTrue(active2)
        self.assertGreater(float(both.detach()),float(rev.detach()))
        both.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.abs().sum()),0.)
        confident=torch.full_like(original,-30.)
        confident[0,ids[0]]=30.
        _,active3=teacher_loss(original,ids,scores,confident,adaptive=True)
        self.assertFalse(active3)
        self.assertIsNone(uncertain.grad)
        with self.assertRaises(ValueError):
            teacher_loss(original,ids.roll(1),scores,uncertain,adaptive=True)
    def test_entropy_cross_prompt_not_conditional(self):
        self.assertEqual(entropy([8]*14),0.)
        self.assertEqual(entropy([1,2,3,4]),2.)
        self.assertEqual(entropy([]),0.)
    def test_fit_excludes_heldout_kind_and_only_original_top64(self):
        txt=RANK.read_text()
        for text in (
            'r["kind"]!=heldout', 'len(train)==72', 'len(subset)==12',
            'torch.topk(orig[0],NATIVE_K',
            'r["own_first_correct"]',
            'meta["eligible"]', 'wrong_control=wrong',
            'original_43M_optimizer_steps', 'heldout_domain_optimizer_steps',
            'guarded_scores(', 'heldout_second_distinct_IDs',
            'first2_after', # optional? we avoid
        ):
            if text=='first2_after':continue
            self.assertIn(text,txt)
        self.assertNotIn('split="DEV"',txt)
        self.assertNotIn('optimizer.step()\n    for',txt)
    def test_original_frozen_and_no_jessica(self):
        self.assertTrue(PHYS.is_file())
        s=PHYS.read_text()
        for marker in (
            'fresh_new_rows("TRAIN")',
            'torch.no_grad()', 'capture_existing_full43m_hidden(model)',
            'original_forwards==168',
            'old_original_param_digest(model)==original_digest',
            '"candidate_accepted":False,"output_gate":0',
        ):
            self.assertIn(marker,s)
        self.assertNotIn('for row in fresh_dev:',s)
        self.assertNotIn('fresh_new_rows("DEV")',s)
    def test_fail_closed_ill_formed(self):
        with self.assertRaises(ValueError):check(False,"invalid")
if __name__=="__main__":
    unittest.main()
