"""S78 synthetic original43M H8 ABI, eight independent mini-head source tests.

This does not replace physical original S75 source-only TRAIN90/30 CI.
"""
import ast
import unittest
from pathlib import Path

import torch

from research.nra3_s20.s78_train_true43m_quote_mention_8_specialists import (
    S78MentionSituationSpecialists,N_PAID,AREAS,STAGE,scores,
)


class TestS78True43MMentionSpecialists(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7878)
        self.model=S78MentionSituationSpecialists()
        self.x=torch.randn(120,6,339,dtype=torch.float32)
        self.x[:,:,320:339]=0
        self.x[:,1,338]=1.
        self.valid=torch.ones(120,6,dtype=torch.bool)
        self.idx=torch.randint(0,6,(120,8),dtype=torch.long)
        self.allowed=torch.ones(120,8,dtype=torch.bool)

    def test_eight_independent_trainable_specialists_and_physical_budget(self):
        budget=self.model.budget()
        self.assertEqual(len(self.model.areas),len(AREAS))
        self.assertEqual(len(AREAS),8)
        self.assertEqual(budget["new_FP16_trainable_parameters"],7496)
        self.assertEqual(budget["new_FP16_weight_bytes"],14992)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),N_PAID)
        self.assertEqual(len({id(z.net[0].weight) for z in self.model.areas}),8)
        self.assertEqual(budget["output_gate"],0)
        self.assertFalse(budget["activation_allowed"])
        self.assertFalse(budget["candidate_accepted"])

    def test_forward_source_only_all_proposals_and_gradient_no_old_update(self):
        logit=self.model(self.x,self.valid,self.idx,self.allowed)
        self.assertEqual(logit.shape,(120,8,8))
        self.assertTrue(torch.isfinite(logit).all())
        loss=logit[:,0].square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and
                            bool(torch.isfinite(p.grad).all())
                            for p in self.model.parameters()))
        self.assertIsNone(self.x.grad)
        self.assertTrue(torch.isfinite(logit).all())

    def test_quote_word_in_new_source_router_but_no_assertion(self):
        src=self.x.clone()
        self.idx[:,0]=1
        with torch.no_grad():
            with_quote=self.model(src,self.valid,self.idx,self.allowed)
            src[:,1,338]=0.
            without_quote=self.model(src,self.valid,self.idx,self.allowed)
        self.assertFalse(torch.equal(with_quote[:,0],without_quote[:,0]))
        self.assertEqual(self.model.output_gate,0)
        self.assertFalse(self.model.activation_allowed)

    def test_gate_fails_closed_even_if_attempt_to_activate(self):
        self.model.output_gate=1
        with self.assertRaisesRegex(ValueError,"no output activation"):
            self.model(self.x,self.valid,self.idx,self.allowed)

    def test_no_constructed_gold_top_level_import_before_source_forward(self):
        tree=ast.parse(Path("research/nra3_s20/s78_train_true43m_quote_mention_8_specialists.py").read_text())
        imports=[z.module for z in tree.body if isinstance(z,ast.ImportFrom)]
        self.assertTrue(all("TRAIN_LOSS_ONLY" not in (x or "") for x in imports))
        source=next(z for z in tree.body if isinstance(z,ast.FunctionDef)
                    and z.name=="source_first_then_train_gold")
        self.assertTrue(any(isinstance(x,ast.ImportFrom) and
                            "TRAIN_LOSS_ONLY" in (x.module or "")
                            for x in ast.walk(source)))

    def test_gold_event_anchor_ONLY_in_metric_posthoc_not_forward(self):
        slot=torch.zeros(120,dtype=torch.long)
        label=torch.zeros(120,8)
        label[:,0]=1.
        subset=torch.zeros(120,dtype=torch.bool)
        subset[:30]=True
        r=scores(self.model,self.x,self.valid,self.idx,self.allowed,
                 slot,label,subset)
        self.assertEqual(r["n_constructed_TRAIN_event_anchor_in_source_only_mention_beam"],30)
        self.assertTrue(r["gold_event_anchor_used_ONLY_for_teacher_forced_diagnostic"])
        self.assertTrue(r["model_forward_without_gold_labels"])
        self.assertTrue(r["not_autonomous_mention_factuality_or_recipient_detection"])
        self.assertTrue(0<=r["micro_f1"]<=1)

if __name__=="__main__":
    unittest.main()
