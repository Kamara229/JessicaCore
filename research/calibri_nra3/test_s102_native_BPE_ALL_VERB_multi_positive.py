"""NRA3 S102: synthetic ONLY native-BPE all-verb/multi-edge ABI tests.

No real third-party UD Taiga labels, no original616 checkpoints,
no sealed prospective TRAIN-c. Confirms labels never feed neural
forward, unknown never becomes negative, all verb/role slots retained.
"""
from pathlib import Path
from unittest.mock import patch
import ast
import unittest

import torch

from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,UD_CASE,UD_UPOS,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    sha256_serialized,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    offline_all_verb_multi_positive_relation_loss,
)
from research.calibri_nra3 import s102_real12_TaigaTRAINd_multiverb_gold_AFTER_sourceSHA as s102
from research.calibri_nra3.s102_TaigaD_real12_multiverb_28887_neural_train import (
    SEED,EPOCHS,LR,PAID,
    batches_fixed,select_targets,
)


def constructed():
    raw="Иван передал письмо Марии и убежал"
    spans=[[0,4],[5,12],[13,19],[20,25],[26,27],[28,34]]
    source=[{"row_id":f"TEST_S100_TAIGAD_{i:03d}","text":raw}
            for i in range(256)]
    gold=[]
    for i,z in enumerate(source):
        gold.append({
            "row_id":z["row_id"],
            "complete_original_source_UD_word_start_end_chars_TRAIN_ONLY":
                [list(x) for x in spans],
            "all_UD_VERB_predicates_and_all_unambiguous_multi_role_edges_TRAIN_ONLY":[
                {"predicate_UD_word_zero_index_TRAIN_ONLY":1,
                 "multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY":[
                    {"role":"actor","arg_ud_word":0},
                    {"role":"patient","arg_ud_word":2},
                    {"role":"patient","arg_ud_word":3}]},
                {"predicate_UD_word_zero_index_TRAIN_ONLY":5,
                 "multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY":[
                    {"role":"actor","arg_ud_word":0},
                    {"role":"event_relation","arg_ud_word":1}]},
            ],
            "reported_speech_factuality_coreference_not_adjudicated_UNKNOWN":True,
            "no_negative_event_labels_invented":True,
            "multiple_distinct_arguments_PER_role_NOT_collapsed":True,
            "original_Calibri_BPE_alignment_NOT_YET_known":True,
            "offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY":
                ["NOUN","VERB","NOUN","PROPN","CCONJ","VERB"],
            "offline_Case_ALL_UD_WORDS_TRAIN_ONLY":
                ["Nom",None,"Acc","Dat",None,None],
            "offline_Number_ALL_UD_WORDS_TRAIN_ONLY":
                ["Sing","Sing","Sing","Sing",None,"Sing"],
            "offline_Voice_ALL_UD_WORDS_TRAIN_ONLY":
                [None,"Act",None,None,None,"Act"],
        })
    indexes=torch.tensor([0,63,64,128])
    start=torch.tensor([[0,5,13,20,26,28]]*4,dtype=torch.int32)
    end=torch.tensor([[4,12,19,25,27,34]]*4,dtype=torch.int32)
    last=torch.tensor([[1,2,3,4,5,6]]*4,dtype=torch.int32)
    safe=torch.ones((4,6),dtype=torch.bool)
    valid=torch.tensor([[True]*8+[False]]*4,dtype=torch.bool)
    return source,gold,indexes,start,end,last,safe,valid


def aligned(modify=None):
    args=constructed()
    if modify:modify(*args)
    src,gold,ix,a,b,c,d,valid=args
    with patch.object(s102,"SOURCE_SHA",sha256_serialized(src)),\
         patch.object(s102,"GOLD_SHA",sha256_serialized(gold)):
        return s102.map_UD_after_original_source_sha(
            source=src,gold=gold,valid=valid,original_indices=ix,
            raw_start=a,raw_end=b,raw_final_BPE=c,raw_safe=d)


class TestS102AllVerbNativeBPE(unittest.TestCase):
    def test_all_verb_mentioned_slots_and_multiple_positive_args_not_collapsed(self):
        g=aligned()
        self.assertEqual(g.counts["full_UD_BPE_rows"],4)
        self.assertEqual(g.counts["known_verb_predicate_BPE_positions"],8)
        self.assertEqual(g.counts["positive_event_argument_edges"],20)
        self.assertEqual(g.train.tolist(),[True,False,True,True])
        self.assertEqual(g.internal_dev.tolist(),[False,True,False,False])
        role=ROLE_NAMES.index("patient")
        self.assertTrue(g.role.event_role_supervised[0,role,2])
        self.assertTrue(g.role.argument_gold_positive[0,role,2,3])
        self.assertTrue(g.role.argument_gold_positive[0,role,2,4])
        self.assertEqual(int(g.role.argument_gold_positive[0,role,2].sum()),2)
        self.assertTrue(g.role.event_verb_positive_only[0,2])
        self.assertTrue(g.role.event_verb_positive_only[0,6])

    def test_new_nra3_morphology_targets_known_and_unannotated_unknown(self):
        g=aligned()
        self.assertEqual(int(g.morphology["upos"][0,1]),UD_UPOS.index("NOUN"))
        self.assertEqual(int(g.morphology["upos"][0,2]),UD_UPOS.index("VERB"))
        self.assertEqual(int(g.morphology["case"][0,1]),UD_CASE.index("Nom"))
        self.assertEqual(int(g.morphology["case"][0,3]),UD_CASE.index("Acc"))
        self.assertEqual(int(g.morphology["case"][0,2]),s102.UNKNOWN)
        self.assertEqual(int(g.morphology["upos"][0,0]),s102.UNKNOWN)

    def test_unsafe_argument_word_rejects_whole_row_no_ghost_training(self):
        def unsafe(src,gold,idx,a,b,c,safe,valid):
            safe[1,2]=False
        g=aligned(unsafe)
        self.assertEqual(g.eligible.tolist(),[True,False,True,True])
        self.assertEqual(len(g.excluded),1)
        self.assertTrue((g.morphology["upos"][1]==s102.UNKNOWN).all())
        self.assertFalse(bool(g.role.argument_gold_positive[1].any()))

    def test_invalid_duplicate_predicate_or_unmapped_verb_rejects_all(self):
        def bad(src,gold,idx,a,b,c,safe,valid):
            gold[0]["all_UD_VERB_predicates_and_all_unambiguous_multi_role_edges_TRAIN_ONLY"][1][
                "predicate_UD_word_zero_index_TRAIN_ONLY"]=1
        g=aligned(bad)
        self.assertEqual(g.eligible.tolist(),[False,True,True,True])
        self.assertFalse(bool(g.role.event_role_supervised[0].any()))

    def test_original_BPE_multi_positive_neural_loss_has_grads_to_in_calibri(self):
        g=aligned()
        net=NRA3MorphologyInCalibriWeights()
        graph=net(torch.randn(4,12,9,320),g.role.native_BPE_valid)
        loss,parts=offline_all_verb_multi_positive_relation_loss(graph,g.role)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["n_source_BPE_verbs_explicit"],8)
        self.assertEqual(parts["n_supervised_event_to_argument_edges"],20)
        loss.backward()
        self.assertIsNotNone(net.role_bilinear.grad)
        self.assertTrue(torch.isfinite(net.role_bilinear.grad).all())
        self.assertEqual(net.output_gate,0)

    def test_fixed_source_stratum_train_batches_NEVER_use_internal(self):
        ix=torch.arange(256)
        mask=ix%64<48
        b=batches_fixed(ix,mask)
        self.assertEqual(len(b),24)
        union=torch.cat(b)
        self.assertEqual(len(union),192)
        self.assertEqual(len(set(union.tolist())),192)
        self.assertFalse(bool((union%64>=48).any()))

    def test_checkpoint_and_full_meta_neural_parent_provenance_fixed(self):
        self.assertEqual(SEED,1021119)
        self.assertEqual(EPOCHS,26)
        self.assertEqual(LR,0.0007)
        self.assertEqual(PAID,28887)
        self.assertEqual(s102.SOURCE_SHA,
            "bc12818689e147f9fd167838bd22e220fd9a28b46bf51ccd750fa61ed4ddafc3")
        self.assertEqual(s102.GOLD_SHA,
            "c96f5ef54982c5ed486d725eb0de19f1f3b81bde1b080ad12a6208e6eaa090c7")

    def test_no_UD_gold_at_inference_and_no_S96_S99_dev_or_train_c_import(self):
        p=Path("research/calibri_nra3/s97_in_calibri_morphological_12layer_neural_graph.py")
        tree=ast.parse(p.read_text())
        imports=[z.module or "" for z in ast.walk(tree)
                 if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s102" in z or "s100_new" in z
                             or "s96" in z or "s99" in z
                             or "train_c" in z.lower() for z in imports))
        self.assertNotIn("offline_all_verb_multi_positive_relation_loss",
                         p.read_text())


if __name__=="__main__":
    unittest.main()
