"""S107: source-only synthetic ABI for rank-competitive NRA3, no old DEV or GOLD."""
from __future__ import annotations
import ast
from dataclasses import replace
from pathlib import Path
import unittest

import torch
from torch import nn
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,STAGE,
    NEW_NEURAL_PARAMETERS,TOTAL_NEURAL_PARAMETERS,NEW_KEYS,
    OLD_NEURAL_PARAMETERS,PAIR_INPUT,STRUCTURAL_DIM,
)

class Model12(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(12)])


class TestS107InCalibriLearnedPredicateArgumentCompetition(unittest.TestCase):
    def test_53743_in_calibri_params_preserve_all8_NRA2_and_old_memory(self):
        m=NRA3StructuralCompetitionInCalibriWeights()
        b=m.budget()
        self.assertIn("S107",STAGE)
        self.assertEqual((NEW_NEURAL_PARAMETERS,OLD_NEURAL_PARAMETERS,
                          TOTAL_NEURAL_PARAMETERS),(16656,37087,53743))
        self.assertEqual(PAIR_INPUT,128)
        self.assertEqual(STRUCTURAL_DIM,32)
        self.assertEqual(b["nominal_FP16_weight_bytes"],107486)
        self.assertEqual(b["NRA2_independent_mini_areas_preserved"],8)
        self.assertEqual(b["trainable_in_model_memory_slots"],16)
        self.assertEqual(b["all_12_original_calibri_layers"],12)
        self.assertEqual(tuple(m.structural_role_bilinear.shape),(8,32,32))
        self.assertEqual(tuple(m.structural_native_BPE_distance.weight.shape),(33,8))
        self.assertTrue(torch.equal(
            m.structural_competition_gate,torch.zeros(8)))
        self.assertFalse(m.candidate_accepted)
        self.assertFalse(m.activation_allowed)
        self.assertEqual(m.output_gate,0)

    def test_new_zero_gate_REALLY_bitidentical_to_S105_37087_old_all_logits(self):
        torch.manual_seed(107001)
        old=NRA3RelationalMorphologyInCalibriWeights().eval()
        torch.manual_seed(107001)
        new=NRA3StructuralCompetitionInCalibriWeights().eval()
        a=old.state_dict();b=new.state_dict()
        self.assertTrue(set(a)<set(b))
        self.assertEqual(set(b)-set(a),NEW_KEYS)
        self.assertTrue(all(torch.equal(v,b[k]) for k,v in a.items()))
        x=torch.randn(2,12,7,320)
        valid=torch.tensor([[True]*7,[True]*5+[False]*2])
        with torch.inference_mode():
            p=old(x,valid)
            q=new(x,valid)
        for name in ("event_mention_logits","role_arc_logits",
                     "event_state_logits","mini_area_logits",
                     "layer_router_weights","memory_slot_attention",
                     "upos_logits","case_logits","number_logits","voice_logits"):
            self.assertTrue(torch.equal(getattr(p,name),
                                        getattr(q,name)),name)
        self.assertTrue(bool((q.role_arc_logits[1,:,:,5:]==-1.e4).all()))
        self.assertTrue(bool((q.role_arc_logits[1,:,5:,:]==-1.e4).all()))

    def test_normalized_gate_can_REPLACE_large_old_lexical_rank_margin(self):
        m=NRA3StructuralCompetitionInCalibriWeights()
        a=torch.full((1,8,3,3),-1.e4)
        z=a.clone()
        a[:,:,0,:2]=torch.tensor([8.,0.])
        z[:,:,0,:2]=torch.tensor([0.,8.])
        # Remaining two source predicates valid but have finite signals.
        a[:,:,1,:2]=torch.tensor([6.,0.])
        z[:,:,1,:2]=torch.tensor([0.,6.])
        valid=torch.tensor([[True,True,False]])
        old=m.combine_role_competition(a,z,valid)
        self.assertTrue(torch.equal(old,a))
        with torch.no_grad():
            m.structural_competition_gate.fill_(4.)
        new=m.combine_role_competition(a,z,valid)
        self.assertEqual(int(new[0,0,0].argmax()),1)
        self.assertTrue(torch.equal(new[:,:,:,2],
                                     torch.full_like(new[:,:,:,2],-1.e4)))
        self.assertTrue(bool(torch.isfinite(new).all()))

    def test_nonzero_gate_gives_source_only_structural_pair_gradients(self):
        torch.manual_seed(107002)
        m=NRA3StructuralCompetitionInCalibriWeights()
        with torch.no_grad():
            m.structural_competition_gate.fill_(0.25)
        x=torch.randn(2,12,5,320)
        valid=torch.ones(2,5,dtype=torch.bool)
        y=m(x,valid)
        loss=(y.role_arc_logits[0,0,1,4]-
              y.role_arc_logits[1,1,2,3])
        loss.backward()
        self.assertGreater(float(m.structural_predicate_query.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.structural_argument_key.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.structural_role_bilinear.grad.abs().sum()),0.)
        self.assertGreater(float(m.structural_competition_gate.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(m.structural_role_bilinear.grad).all())
        self.assertIsNone(x.grad)

    def test_source_features_are_token_permutation_equivariant_before_distance(self):
        torch.manual_seed(107003)
        m=NRA3StructuralCompetitionInCalibriWeights().eval()
        h=torch.randn(1,12,6,320)
        valid=torch.ones(1,6,dtype=torch.bool)
        base=NRA3RelationalMorphologyInCalibriWeights().eval()
        base.load_state_dict({k:v for k,v in m.state_dict().items()
                              if k not in NEW_KEYS},strict=True)
        with torch.inference_mode():
            g=base(h,valid)
            q=m.source_only_pair_features(g,h)
            perm=torch.tensor([5,1,2,0,4,3])
            gperm=replace(
                g,upos_logits=g.upos_logits[:,perm],
                case_logits=g.case_logits[:,perm],
                number_logits=g.number_logits[:,perm],
                voice_logits=g.voice_logits[:,perm],
                native_valid=g.native_valid[:,perm])
            qperm=m.source_only_pair_features(gperm,h[:,:,perm])
        self.assertTrue(torch.allclose(q[:,perm],qperm,atol=1.e-6))
        self.assertTrue(bool(torch.isfinite(q).all()))

    def test_only_new_append_neural_state_registered_inside_original_model(self):
        model=Model12()
        child=NRA3StructuralCompetitionInCalibriWeights()
        child.attach_to_RESEARCH_calibri(model)
        weights={k:v for k,v in model.state_dict().items()
                 if k.startswith("model.nra3_neural_weights.")}
        self.assertEqual(sum(v.numel() for v in weights.values()),53743)
        self.assertEqual(len(weights),len(child.state_dict()))
        self.assertTrue(any("mini_areas" in k for k in weights))
        self.assertTrue(any("event_memory_slots" in k for k in weights))
        self.assertTrue(any("structural_competition_gate" in k for k in weights))

    def test_no_external_parsing_teacher_licenses_or_spent_eval_import_in_forward(self):
        path=Path("research/calibri_nra3/s107_in_calibri_structural_role_competition_53743.py")
        t=ast.parse(path.read_text(encoding="utf8"))
        mods=[n.module or "" for n in ast.walk(t) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(z in k.lower() for k in mods
                             for z in ("train_c","s79_","s86_","udpipe",
                                       "stanza","teacher","gold","s105_train")))
        self.assertIn("structural_native_BPE_distance",path.read_text())
        self.assertIn("self.source_only_morph_features",path.read_text())

if __name__=="__main__":
    unittest.main()
