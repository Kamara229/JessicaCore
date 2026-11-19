"""S103: synthetic contracts — structural neural role affinity stays IN Calibri.

No real UD gold, hidden prospective TRAIN-c, old S102 internal dev or
S79/S86 firstreads consulted. This is NOT evidence of quality.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import unittest

import torch
from torch import nn

from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,
)
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,STAGE,
    MORPH_FEATURES,NEW_AFFINITY_PARAMETERS,NEW_TOTAL_NRA3_PARAMS,
)
from research.calibri_nra3.s103_new_Taiga_TRAINe_allVERB_source_gold_wall import (
    TRAIN_FILE,TRAIN_GIT_BLOB,STAGE as DATA_STAGE,
    all_ud_verbs_roles_with_multiedges,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import UDNode


class Model12(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Linear(320,320,bias=False)
                                        for _ in range(12)])


class TestS103MorphSyntacticNeuralInductiveBias(unittest.TestCase):
    def test_37087_params_eight_NRA2_mini_areas_old_memory_and_all12(self):
        model=NRA3RelationalMorphologyInCalibriWeights()
        b=model.budget()
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S103_MORPHO_SYNTACTIC_ROLE_AFFINITY_IN_CALIBRI_RESEARCH")
        self.assertEqual(b["new_neural_parameters"],NEW_TOTAL_NRA3_PARAMS)
        self.assertEqual(b["nominal_FP16_weight_bytes"],74174)
        self.assertEqual(NEW_TOTAL_NRA3_PARAMS,37087)
        self.assertEqual(NEW_AFFINITY_PARAMETERS,8200)
        self.assertEqual(MORPH_FEATURES,32)
        self.assertEqual(model.role_morph_affinity.shape,(8,32,32))
        self.assertEqual(b["NRA2_independent_mini_areas_preserved"],8)
        self.assertEqual(b["all_12_original_calibri_layers"],12)
        self.assertEqual(b["trainable_in_model_memory_slots"],16)
        self.assertFalse(b["no_grammar_gold_or_external_parser_at_neural_inference"] is False)
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)
        self.assertFalse(model.activation_allowed)

    def test_initial_zero_gate_preserves_exact_existing_neural_arc_and_miniareas(self):
        torch.manual_seed(10371)
        old=NRA3MorphologyInCalibriWeights().eval()
        torch.manual_seed(10371)
        new=NRA3RelationalMorphologyInCalibriWeights().eval()
        hidden=torch.randn(2,12,6,320)
        valid=torch.tensor([[True]*6,[True]*4+[False]*2])
        with torch.inference_mode():
            a=old(hidden,valid)
            b=new(hidden,valid)
        for name in ("event_mention_logits","role_arc_logits",
                     "event_state_logits","mini_area_logits",
                     "memory_slot_attention","layer_router_weights",
                     "upos_logits","case_logits","number_logits","voice_logits"):
            self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)),name)

    def test_structural_correction_permutation_equivariant_not_absolute_token_lookup(self):
        torch.manual_seed(10372)
        m=NRA3RelationalMorphologyInCalibriWeights().eval()
        hidden=torch.randn(1,12,6,320)
        original=m(hidden,torch.ones(1,6,dtype=torch.bool))
        correction=m.morph_role_correction(original)
        perm=torch.tensor([4,1,5,0,2,3])
        permuted=replace(
            original,
            upos_logits=original.upos_logits[:,perm],
            case_logits=original.case_logits[:,perm],
            number_logits=original.number_logits[:,perm],
            voice_logits=original.voice_logits[:,perm],
            native_valid=original.native_valid[:,perm],
            role_arc_logits=original.role_arc_logits[:,:,perm][:,:,:,perm],
        )
        new_corr=m.morph_role_correction(permuted)
        expected=correction[:,:,perm][:,:,:,perm]
        self.assertTrue(torch.allclose(new_corr,expected,atol=3.e-6))

    def test_morph_affinity_is_trainable_with_gate_nonzero_and_no_teacher_in_forward(self):
        torch.manual_seed(10373)
        m=NRA3RelationalMorphologyInCalibriWeights()
        with torch.no_grad():
            m.role_morph_affinity_gate.fill_(0.3)
        hidden=torch.randn(1,12,5,320)
        valid=torch.ones(1,5,dtype=torch.bool)
        graph=m(hidden,valid)
        loss=graph.role_arc_logits[0,0,2,4]
        loss.backward()
        self.assertIsNotNone(m.role_morph_affinity.grad)
        self.assertGreater(float(m.role_morph_affinity.grad.abs().sum()),0.)
        self.assertIsNotNone(m.role_morph_affinity_gate.grad)
        self.assertTrue(torch.isfinite(m.role_morph_affinity.grad).all())
        self.assertIsNone(hidden.grad)
        tree=ast.parse(Path(
            "research/calibri_nra3/s103_in_calibri_morphosyntactic_affinity_37087.py"
        ).read_text(encoding="utf8"))
        imports=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any("s103_new_Taiga" in x or "s102_train" in x
                             or "udpipe" in x.lower() or "stanza" in x.lower()
                             or "train_c" in x for x in imports))

    def test_inherit_child_checkpoint_inside_actual_calibri_weights_not_sidecar(self):
        model=Model12()
        original={k:v.clone() for k,v in model.state_dict().items()}
        child=NRA3RelationalMorphologyInCalibriWeights().attach_to_RESEARCH_calibri(model)
        self.assertIs(model.model.nra3_neural_weights,child)
        state=model.state_dict()
        self.assertEqual(sum(p.numel() for p in model.model.nra3_neural_weights.parameters()),37087)
        self.assertTrue(any(x.startswith("model.nra3_neural_weights.role_morph_affinity")
                            for x in state))
        for k,v in original.items():
            self.assertTrue(torch.equal(state[k],v))

    def test_train_e_is_pinned_NEW_not_reserved_train_c_or_old_train_d(self):
        self.assertEqual(TRAIN_FILE,"ru_taiga-ud-train-e.conllu")
        self.assertEqual(TRAIN_GIT_BLOB,"8df5a8b13b934576aa50b7fb55f8a16cb0151003")
        self.assertEqual(DATA_STAGE,
            "CALIBRI_NRA3_S103_ALL_UD_VERBS_MULTI_EDGE_TAIGA_TRAIN_E_PRECOMMITTED")

    def test_all_verb_role_edges_preserved_from_S100_offline_training_only(self):
        def node(i,w,p,h,rel,feat="_"):
            return UDNode(i,w,w,p,feat,h,rel,"_")
        words=(
            node(1,"Курьер","NOUN",2,"nsubj","Case=Nom"),
            node(2,"передал","VERB",0,"root"),
            node(3,"письмо","NOUN",2,"obj","Case=Acc"),
            node(4,"счёт","NOUN",2,"obj","Case=Acc"),
            node(5,"водителю","NOUN",2,"iobj","Case=Dat"),
        )
        events=all_ud_verbs_roles_with_multiedges(words)
        self.assertEqual(len(events),1)
        edges=events[0]["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]
        self.assertIn({"role":"actor","arg_ud_word":0},edges)
        self.assertIn({"role":"patient","arg_ud_word":2},edges)
        self.assertIn({"role":"patient","arg_ud_word":3},edges)
        self.assertIn({"role":"recipient","arg_ud_word":4},edges)

    def test_fail_closed_when_attempting_to_activate_new_neural_graph(self):
        m=NRA3RelationalMorphologyInCalibriWeights()
        m.output_gate=1
        with self.assertRaisesRegex(ValueError,"activate"):
            m(torch.zeros(1,12,4,320),torch.ones(1,4,dtype=torch.bool))


if __name__=="__main__":
    unittest.main()
