"""S97 research-only neural morphology and pinned train-b source/gold ABI.

Synthetic tests establish that morphology really lives in the same
Calibri state_dict, that the eight NRA2 heads and original S91 base
outputs are unchanged at zero-gate initialization, and that external
UD dependency labels are never inputs to neural forward.
NO grammatical generalization/knowledge retention claimed.
"""
from __future__ import annotations
import ast
import inspect
import unittest
from pathlib import Path

import torch
from torch import nn

from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,ROLE_NAMES,AREAS,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,NRA3MorphologyGraph,
    STAGE,UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)
from research.calibri_nra3.s97_pin_Taiga_TRAINb_morph_source_vs_labels import (
    TRAIN_B_BLOB_SHA,TRAIN_B_FILE,one_feat,
)


class ToyCalibri(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(
            [nn.Linear(320,320,bias=False) for _ in range(12)])

    def forward(self,hidden):
        for layer in self.model.layers:
            hidden=layer(hidden)
        return hidden


class TestS97NeuralMorphologyInWeights(unittest.TestCase):
    def test_twelve_layer_router_eight_NRA2_miniareas_and_new_morphology_heads(self):
        m=NRA3MorphologyInCalibriWeights()
        budget=m.budget()
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S97_MORPHOLOGY_RELATIONAL_IN_CALIBRI_NEURAL_WEIGHTS_GATE0")
        self.assertGreater(budget["new_neural_parameters"],22311)
        self.assertLess(budget["new_neural_parameters"],40000)
        self.assertEqual(budget["all_12_original_calibri_layers"],12)
        self.assertEqual(budget["learned_layer_streams"],4)
        self.assertEqual(budget["NRA2_independent_mini_areas_preserved"],8)
        self.assertEqual(tuple(m.mini_areas),tuple(AREAS))
        self.assertEqual(len(set(id(x[0].weight) for x in m.mini_areas.values())),8)
        self.assertEqual(budget["trainable_in_model_memory_slots"],16)
        self.assertFalse(budget["external_inference_UD_parser_required"])
        self.assertFalse(m.is_trained)
        self.assertFalse(m.candidate_accepted)
        self.assertFalse(m.activation_allowed)
        self.assertEqual(m.output_gate,0)

    def test_original_S91_eight_heads_arcs_and_event_memory_preserved_at_initial_zero_gate(self):
        torch.manual_seed(9700)
        old=NRA3InCalibriNeuralWeights().eval()
        torch.manual_seed(9700)
        new=NRA3MorphologyInCalibriWeights().eval()
        with torch.inference_mode():
            h=torch.randn(2,12,7,320)
            valid=torch.tensor([[True]*7,[True]*5+[False]*2])
            a=old(h,valid)
            b=new(h,valid)
        self.assertTrue(torch.equal(a.event_mention_logits,b.event_mention_logits))
        self.assertTrue(torch.equal(a.role_arc_logits,b.role_arc_logits))
        self.assertTrue(torch.equal(a.event_state_logits,b.event_state_logits))
        self.assertTrue(torch.equal(a.mini_area_logits,b.mini_area_logits))
        self.assertTrue(torch.equal(a.memory_slot_attention,b.memory_slot_attention))
        self.assertTrue(torch.equal(a.layer_router_weights,b.layer_router_weights))
        self.assertEqual(b.upos_logits.shape,(2,7,17))
        self.assertEqual(b.case_logits.shape,(2,7,9))
        self.assertEqual(b.number_logits.shape,(2,7,3))
        self.assertEqual(b.voice_logits.shape,(2,7,3))
        self.assertTrue((b.upos_logits[1,5:]==-1.e4).all())

    def test_all_new_morphological_heads_are_differentiable_and_role_graph_connected(self):
        m=NRA3MorphologyInCalibriWeights()
        h=torch.randn(2,12,5,320)
        g=m(h,torch.ones(2,5,dtype=torch.bool))
        loss=(g.upos_logits.square().mean()
              +g.case_logits.square().mean()
              +g.number_logits.square().mean()
              +g.voice_logits.square().mean()
              +g.role_arc_logits.square().mean())
        loss.backward()
        for name in ("morphology_fuse.weight","morph_upos.weight",
                     "morph_case.weight","morph_number.weight",
                     "morph_voice.weight","morph_role_scale"):
            p=dict(m.named_parameters())[name]
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertIsNone(h.grad)

    def test_new_child_stored_INSIDE_original_Calibri_module_tree_exact_old_weights(self):
        torch.manual_seed(9702)
        m=ToyCalibri().eval()
        before={k:z.detach().clone() for k,z in m.state_dict().items()}
        child=NRA3MorphologyInCalibriWeights().eval()
        child.attach_to_RESEARCH_calibri(m)
        names=list(m.state_dict())
        self.assertTrue(any(k.startswith("model.nra3_neural_weights.morph_upos.")
                            for k in names))
        self.assertTrue(any(k.startswith("model.nra3_neural_weights.mini_areas.")
                            for k in names))
        for k,p in before.items():
            self.assertTrue(torch.equal(p,m.state_dict()[k]),k)
        self.assertIs(m.model.nra3_neural_weights,child)

    def test_all_12_hidden_sources_affect_new_neural_morphology(self):
        torch.manual_seed(9704)
        m=NRA3MorphologyInCalibriWeights().eval()
        h=torch.randn(1,12,5,320)
        v=torch.ones(1,5,dtype=torch.bool)
        with torch.inference_mode():
            reference=m(h,v).case_logits
            first=h.clone();first[:,0]+=1.5
            last=h.clone();last[:,11]-=1.5
            self.assertFalse(torch.equal(reference,m(first,v).case_logits))
            self.assertFalse(torch.equal(reference,m(last,v).case_logits))

    def test_neural_model_forward_accepts_NO_teacher_UD_POS_case_or_role_gold(self):
        sig=inspect.signature(NRA3MorphologyInCalibriWeights.forward)
        self.assertEqual(set(sig.parameters),
                         {"self","original_layer_hidden","source_native_token_valid"})
        p=Path("research/calibri_nra3/s97_in_calibri_morphological_12layer_neural_graph.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        imports=[node.module or "" for node in ast.walk(tree)
                 if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any("ud_taiga" in z.lower()
                             or "s94_pin" in z.lower()
                             or "s96_train" in z.lower()
                             or "s79" in z.lower() or "s86" in z.lower()
                             or "stanza" in z.lower() or "udpipe" in z.lower()
                             for z in imports))
        self.assertEqual(len(UD_UPOS),17)
        self.assertEqual(len(UD_CASE),9)
        self.assertEqual(len(UD_NUMBER),3)
        self.assertEqual(len(UD_VOICE),3)

    def test_offline_Taiga_TRAIN_B_upstream_pin_and_no_dev_test(self):
        self.assertEqual(TRAIN_B_FILE,"ru_taiga-ud-train-b.conllu")
        self.assertEqual(TRAIN_B_BLOB_SHA,
                         "3b0f6947bd0d6ac022d91d5fb17269450a876bb3")
        self.assertEqual(one_feat("Case=Nom|Gender=Masc|Number=Sing",
                                  "Case",UD_CASE),"Nom")
        self.assertEqual(one_feat("Number=Plur|Case=Gen",
                                  "Number",UD_NUMBER),"Plur")
        self.assertEqual(one_feat("Voice=Pass|VerbForm=Part",
                                  "Voice",UD_VOICE),"Pass")
        self.assertIsNone(one_feat("_","Case",UD_CASE))
        self.assertIsNone(one_feat("Gender=Fem","Case",UD_CASE))

    def test_morphology_word_features_must_be_unambiguous_or_unknown(self):
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            one_feat("Case=Nom,Acc","Case",UD_CASE)
        with self.assertRaisesRegex(ValueError,"multiple"):
            one_feat("Case=Nom|Case=Acc","Case",UD_CASE)
        self.assertIsNone(one_feat("Case=Unrecognized","Case",UD_CASE))

    def test_unverified_neural_morphology_gate_fails_closed(self):
        m=NRA3MorphologyInCalibriWeights()
        h=torch.zeros(1,12,4,320)
        v=torch.ones(1,4,dtype=torch.bool)
        m.output_gate=1
        with self.assertRaisesRegex(ValueError,"activate"):
            m(h,v)
        m.output_gate=0
        self.assertIs(m.preserve_original_output(h),h)


if __name__=="__main__":
    unittest.main()
