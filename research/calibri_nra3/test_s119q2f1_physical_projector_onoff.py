"""S119q2f1: failclosed same-trained-child causal projector ON/OFF contracts."""
from __future__ import annotations

import copy
import unittest
import torch
from torch import nn

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    LearnedCausalEarlyProjector,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    temporarily_switch_projection,compare_same_weights_onoff,
    N_ORIGINAL_DECODER_FORWARDS,N_PROBES,
    N_AB_VARIANTS,N_PROJECTION_CONDITIONS,N_ZERO_GATE,
)


class MinimalChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(0.4))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
    def forward(self,x,*,contextual):
        return x[:,6:,-1,:].mean(dim=1)*self.weight
    def trainables(self):
        return [self.weight]


def physical_causal_input():
    torch.manual_seed(998)
    early=torch.randn(1,6,4,320)
    return torch.cat([early,early[:,5:6].expand(-1,6,-1,-1)],dim=1)


def source_trace(*,enabled,ce_adjust=0.):
    kinds=("roles_word_order","negation","transitive_comparison",
           "event_time","arithmetic","paraphrase")
    rows=[]
    for kind in kinds:
        for i in range(5):
            rows.append({
                "kind":kind,
                "source_id":kind+str(i),
                "source_split":"CALIBRATION" if i<3 else "SOURCE_PROBE",
                "target_native_id":100+i,
                "target_position":0 if i<3 else 1,
                "projector_on":enabled,
                "conditional_native_BPE_CE":10.+i+ce_adjust,
                "predicted_native_BPE_id":500 if enabled else 501,
                "correct_native_next_BPE_greedy":False,
            })
    return rows


class S119Q2F1CausalProjectorAblationContracts(unittest.TestCase):
    def setUp(self):
        self.adapter=LearnedCausalEarlyProjector(MinimalChild())

    def test_01_default_learned_projector_unchanged_on(self):
        self.assertIs(self.adapter.projection_enabled,True)

    def test_02_scope_switch_off_reverts_original_virtual_slots_exact(self):
        x=physical_causal_input()
        before=x.detach().clone()
        with temporarily_switch_projection(self.adapter,False):
            y=self.adapter.representation(x,contextual=True)
            self.assertTrue(torch.equal(y,x))
            self.assertEqual(self.adapter.last_projected_H320_norm,0.)
        self.assertIs(self.adapter.projection_enabled,True)
        self.assertTrue(torch.equal(x,before))

    def test_03_on_projector_produces_nonzero_early_only_residual(self):
        x=physical_causal_input()
        with temporarily_switch_projection(self.adapter,True):
            y=self.adapter.representation(x,contextual=True)
        self.assertTrue(torch.equal(y[:,:6],x[:,:6]))
        self.assertGreater(float((y[:,6:]-x[:,6:]).abs().sum()),0.)
        self.assertIs(self.adapter.projection_enabled,True)

    def test_04_on_off_reuse_identical_trained_weights(self):
        x=physical_causal_input()
        saved={k:v.detach().clone() for k,v in self.adapter.state_dict().items()}
        with temporarily_switch_projection(self.adapter,False):
            self.adapter(x,contextual=True)
        with temporarily_switch_projection(self.adapter,True):
            self.adapter(x,contextual=True)
        self.assertTrue(all(torch.equal(v,self.adapter.state_dict()[k])
                            for k,v in saved.items()))

    def test_05_restores_projector_switch_on_raise(self):
        with self.assertRaisesRegex(RuntimeError,"deliberate"):
            with temporarily_switch_projection(self.adapter,False):
                raise RuntimeError("deliberate")
        self.assertIs(self.adapter.projection_enabled,True)

    def test_06_cannot_switch_when_projector_already_off(self):
        self.adapter.projection_enabled=False
        with self.assertRaisesRegex(GateFailure,"predeclared"):
            with temporarily_switch_projection(self.adapter,False):
                pass

    def test_07_denies_unaccepted_candidate_being_promoted(self):
        self.adapter.candidate_accepted=True
        with self.assertRaisesRegex(GateFailure,"predeclared"):
            with temporarily_switch_projection(self.adapter,False):
                pass

    def test_08_rejects_nonnumeric_projection_toggle(self):
        with self.assertRaisesRegex(GateFailure,"predeclared"):
            with temporarily_switch_projection(self.adapter,0):
                pass

    def test_09_exact_30_source_only_pairwise_and_six_domains(self):
        a=source_trace(enabled=True,ce_adjust=-0.4)
        b=source_trace(enabled=False)
        c=compare_same_weights_onoff(a,b)
        self.assertEqual(len(c["per_source_readonly_comparisons"]),30)
        self.assertAlmostEqual(c["ON_minus_OFF_mean_native_CE"],-0.4)
        self.assertEqual(c["ON_lower_native_CE_prompts"],30)
        self.assertEqual(c["ON_higher_native_CE_prompts"],0)
        self.assertEqual(c["native_greedy_predictions_CHANGED"],30)
        self.assertTrue(c["known_same_author_SOURCE_PROBE_not_blind"])

    def test_10_same_weights_but_projector_on_could_be_worse(self):
        c=compare_same_weights_onoff(
            source_trace(enabled=True,ce_adjust=0.2),
            source_trace(enabled=False))
        self.assertGreater(c["ON_minus_OFF_mean_native_CE"],0.)
        self.assertEqual(c["ON_lower_native_CE_prompts"],0)

    def test_11_mismatched_source_ids_or_target_rejected(self):
        a=source_trace(enabled=True)
        b=source_trace(enabled=False)
        b[-1]["source_id"]="unknown_other_source"
        with self.assertRaisesRegex(GateFailure,"same 30"):
            compare_same_weights_onoff(a,b)
        b=source_trace(enabled=False)
        b[0]["target_native_id"]=999
        with self.assertRaisesRegex(GateFailure,"same 30"):
            compare_same_weights_onoff(a,b)

    def test_12_misassigned_projector_tag_rejected(self):
        a=source_trace(enabled=True)
        b=source_trace(enabled=False)
        a[0]["projector_on"]=False
        with self.assertRaisesRegex(GateFailure,"same 30"):
            compare_same_weights_onoff(a,b)

    def test_13_mutating_source_probe_into_TRAIN_rejected(self):
        a=source_trace(enabled=True)
        b=source_trace(enabled=False)
        a[0]["source_split"]="TRAIN"
        with self.assertRaisesRegex(GateFailure,"same 30"):
            compare_same_weights_onoff(a,b)

    def test_14_reject_too_few_existing_source_probes(self):
        a=source_trace(enabled=True)[:-1]
        b=source_trace(enabled=False)[:-1]
        with self.assertRaisesRegex(GateFailure,"same 30"):
            compare_same_weights_onoff(a,b)

    def test_15_exact_physical_decoders_predeclared_both_children(self):
        self.assertEqual(N_PROBES,30)
        self.assertEqual(N_AB_VARIANTS,2)
        self.assertEqual(N_PROJECTION_CONDITIONS,2)
        self.assertEqual(N_ZERO_GATE,2)
        self.assertEqual(N_ORIGINAL_DECODER_FORWARDS,122)


if __name__=="__main__":
    unittest.main()
