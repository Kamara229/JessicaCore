"""Synthetic S112 logit bound, in-model trust and no GOLD contracts."""
import ast
from pathlib import Path
import unittest
import torch
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,
    TOTAL,OLD,NEW,STAGE,MAX_LOGIT_CORRECTION,
)
class TestS112BoundedNeuralCompetition(unittest.TestCase):
    def test_exact_53751_in_model_and_output_gate0(self):
        m=NRA3BoundedStructuralTrustInCalibriWeights()
        self.assertEqual((OLD,NEW,TOTAL),(53743,8,53751))
        self.assertEqual(m.budget()['new_neural_parameters'],TOTAL)
        self.assertEqual(len(m.structural_trust_gate),8)
        self.assertEqual(float(m.structural_trust_gate.abs().max()),0.)
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.candidate_accepted or m.activation_allowed)
        self.assertEqual(MAX_LOGIT_CORRECTION,1.)

    def test_zero_trust_bit_exact_and_high_trust_cannot_flip_large_old_margin(self):
        m=NRA3BoundedStructuralTrustInCalibriWeights()
        valid=torch.ones((1,3),dtype=torch.bool)
        old=torch.tensor([6.,-2.,-3.]).reshape(1,1,1,3).expand(1,8,3,3).contiguous()
        raw=torch.tensor([-100.,100.,10.]).reshape(1,1,1,3).expand(1,8,3,3).contiguous()
        zero=m.bounded_role_competition(old,raw,valid)
        self.assertTrue(torch.equal(zero,old))
        with torch.no_grad():
            m.structural_trust_gate.fill_(10.)
            output=m.bounded_role_competition(old,raw,valid)
        self.assertTrue(bool(((output-old).abs()<=1.00001).all()))
        self.assertTrue(bool((output.argmax(-1)==old.argmax(-1)).all()))
        self.assertGreater(float((output-old).abs().max()),0.)

    def test_trust_zero_gradient_can_learn_and_inherited_graph_stays_in_weights(self):
        m=NRA3BoundedStructuralTrustInCalibriWeights()
        valid=torch.ones((1,3),dtype=torch.bool)
        old=torch.tensor([4.,-1.,-2.]).reshape(1,1,1,3).expand(1,8,3,3).contiguous()
        raw=torch.tensor([-10.,5.,0.]).reshape(1,1,1,3).expand(1,8,3,3).contiguous()
        scored=m.bounded_role_competition(old,raw,valid)
        scored[0,0,0,0].backward()
        self.assertIsNotNone(m.structural_trust_gate.grad)
        self.assertNotEqual(float(m.structural_trust_gate.grad[0]),0.)
        self.assertIn('role_morph_affinity',dict(m.named_parameters()))
        self.assertIn('structural_role_bilinear',dict(m.named_parameters()))
        self.assertIn('event_memory_slots',dict(m.named_parameters()))

    def test_no_offline_teacher_or_spent_devs_imported_by_neural_module(self):
        p=Path('research/calibri_nra3/s112_in_calibri_bounded_structural_trust_53751.py')
        s=p.read_text()
        self.assertIn('NRA3RelationalMorphologyInCalibriWeights.forward(',s)
        self.assertIn('self.structural_role_scores(',s)
        self.assertIn('structural_trust_gate',s)
        self.assertIn('MAX_LOGIT_CORRECTION*torch.tanh(',s)
        imported=[z.module or '' for z in ast.walk(ast.parse(s))
                  if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any('s108_new_GSD' in x or 'train_c' in x.lower()
                             or 'gold' in x.lower() for x in imported))
if __name__=='__main__':
    unittest.main()
