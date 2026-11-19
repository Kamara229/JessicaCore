"""S105 ONE-TIME prospective original12 Calibri UD Taiga-C scoring ABI.

Constructed synthetic tiny source rows only. No real Taiga train-c
GOLD loaded here; never fits S99/S102, never uses S79/S86 firstreads
or S96/S99/S102 spent internal diagnostic performance.
"""
import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    sha256_serialized,
)
from research.calibri_nra3 import s105_TaigaC_ONE_TIME_teacher_labels_AFTER_true12_SHA as s105
from research.calibri_nra3.s105_ONE_TIME_TaigaC_S99_vs_S102_true12_firstread_eval import (
    EXPECTED,STAGE,BOOTSTRAP_SEED,BOOTSTRAP_REPLICATES,
    paired_row_bootstrap,
)


def constructed_without_real_C_gold():
    text="Иван передал письмо Марии и убежал"
    src=[{"row_id":f"NO_REAL_C_TRAIN_LABELS_{i+1:03d}","text":text}
         for i in range(256)]
    gold=[]
    for row in src:
        gold.append({
            "row_id":row["row_id"],
            "complete_original_source_UD_word_start_end_chars_TRAIN_ONLY":[
                [0,4],[5,12],[13,19],[20,25],[26,27],[28,34]],
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
            "offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY":[
                "NOUN","VERB","NOUN","PROPN","CCONJ","VERB"],
            "offline_Case_ALL_UD_WORDS_TRAIN_ONLY":[
                "Nom",None,"Acc","Dat",None,None],
            "offline_Number_ALL_UD_WORDS_TRAIN_ONLY":[
                "Sing","Sing","Sing","Sing",None,"Sing"],
            "offline_Voice_ALL_UD_WORDS_TRAIN_ONLY":[
                None,"Act",None,None,None,"Act"],
        })
    ix=torch.tensor([0,63,64,128])
    starts=torch.tensor([[0,5,13,20,26,28]]*4,dtype=torch.int32)
    ends=torch.tensor([[4,12,19,25,27,34]]*4,dtype=torch.int32)
    last=torch.tensor([[1,2,3,4,5,6]]*4,dtype=torch.int32)
    safe=torch.ones((4,6),dtype=torch.bool)
    valid=torch.tensor([[True]*8+[False]]*4)
    return src,gold,ix,starts,ends,last,safe,valid


def synthetic_mapping(modify=None):
    src,gold,ix,starts,ends,last,safe,valid=constructed_without_real_C_gold()
    if modify is not None:
        modify(src,gold,ix,starts,ends,last,safe,valid)
    with (patch.object(s105,"SOURCE_SHA",sha256_serialized(src)),
          patch.object(s105,"GOLD_SHA",sha256_serialized(gold))):
        return s105.map_UD_after_original_source_sha(
            source=src,gold=gold,valid=valid,
            original_indices=ix,raw_start=starts,raw_end=ends,
            raw_final_BPE=last,raw_safe=safe)


class TestS105OneTimeProspectiveABI(unittest.TestCase):
    def test_frozen_weight_sha_and_prospective_C_gold_sha_precommitted(self):
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S105_PROSPECTIVE_ONE_SHOT_TAIGA_C_S99_VS_S102_FROZEN_NO_TRAIN")
        self.assertEqual(EXPECTED["S99"],
            "c77a6964ae4e02cb3aac98d69978a25cec2b491a9624cc9668b3b307699bff8a")
        self.assertEqual(EXPECTED["S102"],
            "f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df")
        self.assertEqual(s105.SOURCE_SHA,
            "7c960fd28b14449cfb9fa3611576aa21a88dbd6afccbbde4e2a02d015707bcdb")
        self.assertEqual(s105.GOLD_SHA,
            "e039c3c6a8ac2221cea774895c09da5838b3d50e7d71079c9ff3ca0d285827eb")

    def test_one_time_teacher_all_verb_multi_positive_event_only_metrics(self):
        x=synthetic_mapping()
        self.assertEqual(x.eligible.tolist(),[True]*4)
        self.assertEqual(x.counts["known_verb_predicate_BPE_positions"],8)
        self.assertEqual(x.counts["positive_event_argument_edges"],20)
        p=ROLE_NAMES.index("patient")
        self.assertTrue(x.role.argument_gold_positive[0,p,2,3])
        self.assertTrue(x.role.argument_gold_positive[0,p,2,4])
        self.assertEqual(int(x.role.argument_gold_positive[0,p,2].sum()),2)

    def test_one_time_strict_source_original_BPE_unsafe_argument_excluded(self):
        def spoil(src,gold,ix,starts,ends,last,safe,valid):
            safe[1,2]=False
        d=synthetic_mapping(spoil)
        self.assertEqual(d.eligible.tolist(),[True,False,True,True])
        self.assertEqual(len(d.excluded),1)
        self.assertFalse(bool(d.role.argument_gold_positive[1].any()))

    def test_new_gold_mismatch_before_frozen_model_score_blocks(self):
        src,gold,ix,starts,ends,last,safe,valid=constructed_without_real_C_gold()
        with patch.object(s105,"SOURCE_SHA",sha256_serialized(src)):
            with self.assertRaisesRegex(ValueError,"SHA"):
                s105.map_UD_after_original_source_sha(
                    source=src,gold=gold,valid=valid,
                    original_indices=ix,raw_start=starts,raw_end=ends,
                    raw_final_BPE=last,raw_safe=safe)

    def test_predeclared_deterministic_PAIRED_sentence_bootstrap(self):
        n=128
        denom=[2]*n
        a={
            "per_original_SOURCE_ROW_role_correct_counts":[0]*n,
            "per_original_SOURCE_ROW_role_total_counts":denom,
        }
        b={
            "per_original_SOURCE_ROW_role_correct_counts":[1]*n,
            "per_original_SOURCE_ROW_role_total_counts":denom,
        }
        result=paired_row_bootstrap(a,b)
        self.assertEqual(result["n_replicates"],BOOTSTRAP_REPLICATES)
        self.assertEqual(BOOTSTRAP_SEED,1052026)
        self.assertAlmostEqual(result["delta_S102_minus_S99_paired_role_top1"],.5)
        self.assertAlmostEqual(result["lower_95_percentile"],.5)
        self.assertAlmostEqual(result["upper_95_percentile"],.5)

    def test_NO_training_S99_S102_and_C_firstread_permanently_spent(self):
        p=Path("research/calibri_nra3/s105_ONE_TIME_TaigaC_S99_vs_S102_true12_firstread_eval.py")
        a=p.read_text()
        tree=ast.parse(a)
        imports=[x.module or "" for x in ast.walk(tree)
                 if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s79_" in x or "s86_" in x
                             or "s99_REAL12" in x
                             or "s102_TaigaD_real12" in x
                             for x in imports))
        self.assertNotIn("AdamW",a)
        self.assertNotIn(".backward(",a)
        self.assertNotIn(".step(",a)
        self.assertIn("Taiga_C_prospective_FIRSTREAD_BURNED_never_retrain_retune_rerun",a)
        self.assertIn("source_all12_forward_completely_goldblind",a)
        self.assertIn("teacher_event_anchor_ONLY_in_metrics_not_in_neural_forward",a)
        self.assertIn('"output_gate":0',a)
        self.assertIn('"candidate_accepted":False',a)
        self.assertIn('"activation_allowed":False',a)


if __name__=="__main__":
    unittest.main()
