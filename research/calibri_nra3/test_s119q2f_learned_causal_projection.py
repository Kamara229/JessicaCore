"""S119q2f unit/causality contract: trainable rank8 h0..5, NO GOLD."""
from __future__ import annotations

import copy
import unittest
import torch
from torch import nn

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    LearnedCausalEarlyProjector,RANK,EXPECTED_NEW_PARAMETERS,
    TRAIN_ROWS,EPOCHS,ACCUM,EXPECTED_TRAIN_DECODER_FORWARDS,
    EXPECTED_DIAGNOSTIC_FORWARDS,EXPECTED_ACTUAL_DECODER_FORWARDS,
    LR,STAGE,
)


class DummyRealChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.learned_core=nn.Parameter(torch.tensor([0.5]))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
    def forward(self,x,*,contextual):
        # Distinct calculation on projected virtual slots.
        return ((x[:,6:,-1]-x[:,5:6,-1]).mean(dim=1)*
                self.learned_core).reshape(1,320)
    def trainables(self):
        return [self.learned_core]


def early12():
    torch.manual_seed(7412)
    real=torch.randn(1,6,5,320)
    return torch.cat([real,real[:,5:6].expand(-1,6,-1,-1)],dim=1)


class S119Q2FLearnedCausalProjectionContracts(unittest.TestCase):
    def setUp(self):
        self.child=LearnedCausalEarlyProjector(DummyRealChild())

    def test_01_actual_rank8_projector_parameter_overhead(self):
        self.assertEqual(RANK,8)
        self.assertEqual(EXPECTED_NEW_PARAMETERS,5162)
        self.assertEqual(sum(x.numel() for x in (
            self.child.down.weight,self.child.up.weight,
            self.child.mix_logits,self.child.slot_gain)),5162)
        self.assertEqual(sum(x.numel() for x in self.child.trainables()),
                         5163)

    def test_02_original_first_six_real_hidden_states_untouched(self):
        x=early12()
        for ctx in (True,False):
            out=self.child.representation(x,contextual=ctx)
            self.assertEqual(tuple(out.shape),(1,12,5,320))
            self.assertTrue(torch.equal(out[:,:6],x[:,:6]))
            self.assertTrue(torch.isfinite(out).all().item())

    def test_03_projection_truly_changes_early_virtual_slots(self):
        x=early12()
        z=self.child.representation(x,contextual=True)
        self.assertGreater(float(torch.linalg.vector_norm(
            z[:,6:]-x[:,6:])),0.)

    def test_04_matched_local_uses_identical_trainables_different_source(self):
        x=early12()
        a=self.child.representation(x,contextual=True)
        b=self.child.representation(x,contextual=False)
        self.assertFalse(torch.equal(a[:,6:],b[:,6:]))
        self.assertTrue(torch.equal(a[:,:6],b[:,:6]))

    def test_05_no_future_real_h6_h11_allowed(self):
        x=early12()
        x[:,7]+=2.
        with self.assertRaisesRegex(GateFailure,"NO hidden future"):
            self.child.representation(x,contextual=True)

    def test_06_no_shape_or_nan_spoofed_causal_view(self):
        x=early12()
        with self.assertRaisesRegex(GateFailure,"true early original"):
            self.child.representation(x[:,2:],contextual=True)
        bad=x.clone();bad[0,0,0,0]=float("nan")
        with self.assertRaisesRegex(GateFailure,"true early original"):
            self.child.representation(bad,contextual=True)

    def test_07_actual_projector_gradients_reach_all_early_mixer_modules(self):
        x=early12()
        loss=self.child(x,contextual=True).square().sum()
        loss.backward()
        for name in ("down","up"):
            p=getattr(self.child,name).weight
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()),0.)
        self.assertIsNotNone(self.child.mix_logits.grad)
        self.assertGreater(float(self.child.mix_logits.grad.abs().sum()),0.)
        self.assertIsNotNone(self.child.slot_gain.grad)
        self.assertGreater(float(self.child.slot_gain.grad.abs().sum()),0.)

    def test_08_local_ablation_equal_initial_weights_and_budget(self):
        b=copy.deepcopy(self.child)
        self.assertEqual(sum(x.numel() for x in b.trainables()),
                         sum(x.numel() for x in self.child.trainables()))
        self.assertTrue(all(torch.equal(a,c) for a,c in zip(
            b.state_dict().values(),self.child.state_dict().values())))

    def test_09_child_original_output_gate_must_remain_zero(self):
        self.assertEqual(self.child.output_gate,0)
        self.assertFalse(self.child.candidate_accepted)
        self.assertFalse(self.child.activation_allowed)
        self.child.candidate_accepted=True
        with self.assertRaisesRegex(GateFailure,"research output gate0"):
            self.child(early12(),contextual=True)

    def test_10_child_local_gate_never_activates_by_training(self):
        self.assertFalse(self.child.child.candidate_accepted)
        self.assertEqual(self.child.child.output_gate,0)

    def test_11_optimizer_budget_fixed_before_data_metrics(self):
        self.assertEqual(TRAIN_ROWS,96)
        self.assertEqual(EPOCHS,2)
        self.assertEqual(ACCUM,4)
        self.assertEqual(LR,0.0005)
        self.assertEqual(EXPECTED_TRAIN_DECODER_FORWARDS,384)
        self.assertEqual(EXPECTED_DIAGNOSTIC_FORWARDS,120)
        self.assertEqual(EXPECTED_ACTUAL_DECODER_FORWARDS,504)

    def test_12_cannot_sneak_original_input_gradients_into_projector(self):
        original=early12().detach()
        self.assertFalse(original.requires_grad)
        out=self.child(original,contextual=True)
        self.assertTrue(out.requires_grad)
        self.assertFalse(original.requires_grad)

    def test_13_local_contextual_differ_only_in_view_and_child_scope(self):
        x=early12()
        a=self.child(x,contextual=True)
        b=self.child(x,contextual=False)
        self.assertEqual(tuple(a.shape),(1,320))
        self.assertEqual(tuple(b.shape),(1,320))
        self.assertFalse(torch.equal(a,b))

    def test_14_no_acceptance_in_research_stage(self):
        self.assertIn("CAUSAL_H0H5",STAGE)


if __name__=="__main__":
    unittest.main()
