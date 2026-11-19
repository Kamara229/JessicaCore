"""S99 offline ORIGINAL Calibri native-BPE UD morphology alignment/loss ABI.

Only fake source tokens and generated annotation tensors here; not a
naturalistic grammar benchmark, pretrained original model or S96 test.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import ast
import unittest
import torch

from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    sha256_serialized,
)
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    ROLE_NAMES,STATE_NAMES,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,UD_CASE,UD_UPOS,UD_VOICE,UD_NUMBER,
)
from research.calibri_nra3 import s99_nativeBPE_UD_morphology_labels_AFTER_frozen_original as s99
from research.calibri_nra3.s99_offline_morphology_native_BPE_TRAIN_loss import (
    offline_native_morphology_supervision_loss,UNKNOWN,
)


def source_fixture():
    src=[{"row_id":f"S97_TAIGA_TRAINB_{i+1:03d}",
          "text":f"Инженер передал письмо {i}"}
         for i in range(256)]
    gold=[]
    for i,row in enumerate(src):
        roles=[-1]*8
        roles[ROLE_NAMES.index("actor")]=0
        roles[ROLE_NAMES.index("patient")]=2
        gold.append({
            "row_id":row["row_id"],
            "not_yet_original_native_BPE_coordinates":True,
            "reported_speech_source_and_coreference_GOLD_UNKNOWN":True,
            "not_independently_adjudicated_for_event_F1":True,
            "word_surface_start_end_char_offsets_TRAIN_ONLY":[
                [0,7],[8,15],[16,22]],
            "predicate_UD_word_zero_index_TRAIN_ONLY":1,
            "role_argument_UD_word_zero_indices_TRAIN_ONLY":roles,
            "UD_voice_passive_TRAIN_ONLY":64<=i<128,
            "UD_negation_particle_link_present_TRAIN_ONLY":128<=i<192,
            "offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY":["NOUN","VERB","NOUN"],
            "offline_Case_ALL_UD_WORDS_TRAIN_ONLY":["Nom",None,"Acc"],
            "offline_Number_ALL_UD_WORDS_TRAIN_ONLY":["Sing","Sing","Sing"],
            "offline_Voice_ALL_UD_WORDS_TRAIN_ONLY":[
                None,"Pass" if 64<=i<128 else "Act",None],
        })
    row_ids=torch.tensor([0,63,64,128],dtype=torch.int64)
    starts=torch.tensor([[0,8,16,-1]]*4,dtype=torch.int32)
    ends=torch.tensor([[7,15,22,-1]]*4,dtype=torch.int32)
    last=torch.tensor([[1,2,3,-1]]*4,dtype=torch.int32)
    safe=torch.tensor([[True,True,True,False]]*4,dtype=torch.bool)
    valid=torch.tensor([[True]*5+[False]*2]*4,dtype=torch.bool)
    return src,gold,row_ids,starts,ends,last,safe,valid


def mapping(mutator=None):
    tensors=source_fixture()
    if mutator: mutator(*tensors)
    src,gold,ids,start,end,last,safe,valid=tensors
    with patch.object(s99,"S94_SOURCE_SHA",sha256_serialized(src)):
        return s99.map_UD_word_labels_ONLY_AFTER_source_features(
            source_rows=src,ud_labels=gold,
            source_valid=valid,source_row_ids=ids,
            raw_char_starts=start,raw_char_ends=end,
            raw_last_original_BPE=last,raw_causal_safe=safe,
            ud_gold_sha_expected=sha256_serialized(gold))


class TestS99RealSourceThenOfflineMorphology(unittest.TestCase):
    def test_original_BPE_UD_actor_patient_and_all_morph_heads_exact(self):
        data=mapping()
        self.assertEqual(data.original_source_indices.tolist(),[0,63,64,128])
        self.assertEqual(data.eligible.tolist(),[True,True,True,True])
        self.assertEqual(data.train.tolist(),[True,False,True,True])
        self.assertEqual(data.same_corpus_development.tolist(),
                         [False,True,False,False])
        self.assertEqual(data.n_mapped_roles,8)
        self.assertEqual(int(data.targets.role_argument_token[0,0,2]),1)
        self.assertEqual(int(data.targets.role_argument_token[0,1,2]),3)
        self.assertEqual(int(data.targets.mention[0,2]),1)
        self.assertEqual(int(data.upos[0,1]),UD_UPOS.index("NOUN"))
        self.assertEqual(int(data.upos[0,2]),UD_UPOS.index("VERB"))
        self.assertEqual(int(data.case[0,1]),UD_CASE.index("Nom"))
        self.assertEqual(int(data.case[0,3]),UD_CASE.index("Acc"))
        self.assertEqual(int(data.number[0,1]),UD_NUMBER.index("Sing"))
        self.assertEqual(int(data.voice[2,2]),UD_VOICE.index("Pass"))
        self.assertEqual(int(data.upos[0,0]),UNKNOWN)
        self.assertEqual(int(data.case[0,2]),UNKNOWN)

    def test_old_eight_NRA2_areas_still_separately_annotated_passive_and_negation(self):
        data=mapping()
        self.assertEqual(float(data.targets.state[2,2,STATE_NAMES.index("passive")]),1.)
        self.assertEqual(float(data.targets.mini_area[2,2,2]),1.)
        self.assertEqual(float(data.targets.state[3,2,STATE_NAMES.index("negated")]),1.)
        self.assertEqual(float(data.targets.mini_area[3,2,1]),1.)
        self.assertEqual(float(data.targets.mini_area[0,2,0]),float(-1))
        self.assertEqual(float(data.targets.state[0,2,STATE_NAMES.index("reported")]),float(-1))

    def test_unaligned_original_BPE_event_source_is_rejected_and_all_heads_ignored(self):
        def unsafe(src,gold,ids,start,end,last,safe,valid):
            safe[2,1]=False
        d=mapping(unsafe)
        self.assertEqual(d.eligible.tolist(),[True,True,False,True])
        self.assertEqual(len(d.row_exclusions),1)
        self.assertEqual(d.row_exclusions[0]["original_source_index"],64)
        self.assertEqual(int(d.targets.mention[2,2]),UNKNOWN)
        self.assertTrue((d.upos[2]==UNKNOWN).all())
        self.assertTrue((d.case[2]==UNKNOWN).all())
        self.assertTrue((d.number[2]==UNKNOWN).all())
        self.assertTrue((d.voice[2]==UNKNOWN).all())

    def test_inaccurate_word_surface_spans_never_guessed_or_used_for_training(self):
        def error(src,gold,ids,start,end,last,safe,valid):
            gold[63]["word_surface_start_end_char_offsets_TRAIN_ONLY"][1]=[8,14]
        d=mapping(error)
        self.assertEqual(d.eligible.tolist(),[True,False,True,True])
        self.assertTrue((d.upos[1]==UNKNOWN).all())
        self.assertEqual(int(d.targets.mention[1,2]),UNKNOWN)

    def test_UD_GOLD_SHA_mismatch_rejected_before_any_new_label(self):
        src,gold,ids,start,end,last,safe,valid=source_fixture()
        with patch.object(s99,"S94_SOURCE_SHA",sha256_serialized(src)):
            with self.assertRaisesRegex(ValueError,"SHA"):
                s99.map_UD_word_labels_ONLY_AFTER_source_features(
                    source_rows=src,ud_labels=gold,
                    source_valid=valid,source_row_ids=ids,
                    raw_char_starts=start,raw_char_ends=end,
                    raw_last_original_BPE=last,raw_causal_safe=safe,
                    ud_gold_sha_expected="0"*64)

    def test_explicit_ONLY_known_pos_case_number_voice_loss_gradient_to_IN_model_neural(self):
        data=mapping()
        torch.manual_seed(9904)
        neural=NRA3MorphologyInCalibriWeights()
        hidden=torch.randn(4,12,7,320)
        graph=neural(hidden,data.targets.original_native_valid)
        loss,parts=offline_native_morphology_supervision_loss(
            graph,data.upos,data.case,data.number,data.voice,
            data.targets.original_native_valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["n_explicitly_annotated_original_BPE_by_head"]["upos"],12)
        self.assertEqual(parts["n_explicitly_annotated_original_BPE_by_head"]["case"],8)
        self.assertEqual(parts["n_explicitly_annotated_original_BPE_by_head"]["number"],12)
        self.assertEqual(parts["n_explicitly_annotated_original_BPE_by_head"]["voice"],4)
        self.assertTrue(parts["unannotated_morphology_not_assumed_negative"])
        loss.backward()
        self.assertIsNotNone(neural.layer_router_logits.grad)
        self.assertIsNotNone(neural.morphology_fuse.weight.grad)
        self.assertIsNotNone(neural.morph_upos.weight.grad)
        self.assertIsNotNone(neural.morph_case.weight.grad)
        self.assertIsNotNone(neural.morph_number.weight.grad)
        self.assertIsNotNone(neural.morph_voice.weight.grad)
        self.assertIsNone(hidden.grad)

    def test_morphology_on_padding_or_illegal_label_fails_closed(self):
        data=mapping()
        torch.manual_seed(9905)
        neural=NRA3MorphologyInCalibriWeights()
        h=torch.randn(4,12,7,320)
        graph=neural(h,data.targets.original_native_valid)
        pos=data.upos.clone()
        pos[0,-1]=UD_UPOS.index("VERB")
        with self.assertRaisesRegex(ValueError,"explicit morph annotation"):
            offline_native_morphology_supervision_loss(
                graph,pos,data.case,data.number,data.voice,
                data.targets.original_native_valid)
        wrong=data.case.clone();wrong[0,1]=999
        with self.assertRaisesRegex(ValueError,"explicit morph annotation"):
            offline_native_morphology_supervision_loss(
                graph,data.upos,wrong,data.number,data.voice,
                data.targets.original_native_valid)

    def test_neural_forward_contains_NO_teacher_loss_or_spent_TEST_imports(self):
        p=Path("research/calibri_nra3/s97_in_calibri_morphological_12layer_neural_graph.py")
        s=ast.parse(p.read_text())
        imports=[z.module or "" for z in ast.walk(s)
                 if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s99" in x.lower()
                             or "s97_pin" in x.lower()
                             or "s96_train" in x.lower()
                             or "s79" in x.lower() or "s86" in x.lower()
                             for x in imports))
        self.assertNotIn("offline_native_morphology_supervision_loss",
                         p.read_text())


if __name__=="__main__":
    unittest.main()
