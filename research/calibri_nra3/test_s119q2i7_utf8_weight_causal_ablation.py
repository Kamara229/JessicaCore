"""S119q2i7 true original43M weight-1x control vs prior-3x contracts."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest import TestCase, main

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i6_fresh_ru_utf8_curriculum import (
    rows as new_ru_rows,verify as source_provenance,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_ru_rows,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    native_train_weights,CONTINUATION_WEIGHT,
)
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,first_divergence,
)
from research.calibri_nra3.s119q2i7_real43m_utf8_weight_causal_ablation import (
    SOURCE_SIZES,CONDITIONS,PRIOR_WEIGHT,CONTROL_WEIGHT,
    TRAIN_PARAMS,ROUNDS,NEW_TRAIN,REHEARSAL_TRAIN,TOTAL_TRAIN,
    STAGE,aggregate_own,scored_utf8_own,
    physical_weighted_q2i6_child,physical_q2i6_gate,
    SEED,
)


class MinimalNative:
    eos_token_id=2
    all_special_ids=[2]
    def convert_ids_to_tokens(self,idx):
        return {155:"Ð",201:"Ĳ",247:"Ŀ",30:".",2:"<|im_end|>",
                37:"5"}.get(idx,".")
    def decode(self,ids,skip_special_tokens=True):
        # Enough native-byte fidelity for test: explicit byte D0 BPE pieces.
        b=bytearray()
        for i in ids:
            p=self.convert_ids_to_tokens(i)
            if i==2 and skip_special_tokens:
                continue
            if p=="Ð":b.append(0xD0)
            elif p=="Ĳ":b.append(0x90)
            elif p=="Ŀ":b.append(0x9D)
            else:b.extend(p.encode("utf8"))
        return bytes(b).decode("utf8",errors="replace")


def sample_scored(tok,i,*,ids=None,answer=None):
    if ids is None:ids=[155,2] if i%2==0 else [155,247,30,2]
    if answer is None:answer=[155,201,2]
    entry={"id":"source"+str(i),"kind":"cyrillic_letter",
           "answer_native_with_eos":answer}
    report={
        "source_id":entry["id"],
        "variant":"GT_RECURRENT",
        "generated_own_native_BPE_ids":ids,
        "decoded_text":tok.decode(ids,skip_special_tokens=True),
        "native_EOS":ids[-1]==tok.eos_token_id,
        "strict_finished_full_answer_exact":ids==answer,
        "text_exact_even_if_not_finished":ids==answer,
        "original_full_decoder_calls":len(ids),
    }
    return scored_utf8_own(tok,report,entry)


class S119Q2I7CausalNativeWeightContracts(TestCase):
    def setUp(self):
        self.tok=MinimalNative()

    def test_01_predeclared_only_one_causal_weight_difference(self):
        self.assertEqual(CONTROL_WEIGHT,1.)
        self.assertEqual(PRIOR_WEIGHT,3.)
        self.assertEqual(CONTINUATION_WEIGHT,3.)
        self.assertEqual(CONDITIONS,("OLD_PHYSICAL_3X","SAME_START_UNIFORM_1X"))

    def test_02_new_1x_old_3x_equal_source_and_budget(self):
        self.assertEqual(SOURCE_SIZES,(18,6,6,6))
        self.assertEqual(NEW_TRAIN,18)
        self.assertEqual(REHEARSAL_TRAIN,6)
        self.assertEqual(TOTAL_TRAIN,24)
        self.assertEqual(ROUNDS,3)
        self.assertEqual(TOTAL_TRAIN*ROUNDS,72)
        self.assertEqual(TRAIN_PARAMS,368051)

    def test_03_no_previous_scored_DEV_into_new_TRAIN(self):
        old_reh=[x for x in old_ru_rows("TRAIN")
                 if x["id"].rsplit("_",1)[-1]=="00"]
        old_dev=[x for x in old_ru_rows("DEV")
                 if x["id"].rsplit("_",1)[-1]=="06"]
        a=new_ru_rows("TRAIN")+old_reh
        b=new_ru_rows("DEV")+old_dev
        self.assertEqual(len(a),24)
        self.assertEqual(len(b),12)
        self.assertTrue(set(x["prompt"] for x in a).isdisjoint(
            x["prompt"] for x in b))

    def test_04_new_author_source_SHA_and_independence_warning(self):
        p=source_provenance()
        self.assertEqual(p["fresh_original_author_TRAIN"],18)
        self.assertEqual(p["fresh_original_author_DEV"],6)
        self.assertTrue(p["DEV_is_small_inspectable_NOT_independent_BLIND"])

    def test_05_3x_default_preserved_for_old_experiment(self):
        w,st=native_train_weights(self.tok,[155,201,2])
        self.assertEqual(w,[1.,3.,1.])
        self.assertEqual(st[1],"INCOMPLETE_UTF8_NEEDS_CONTINUATION")

    def test_06_new_counterfactual_1x_is_uniform_even_utf8_mid_char(self):
        w,st=native_train_weights(
            self.tok,[155,201,2],continuation_weight=1.)
        self.assertEqual(w,[1.,1.,1.])
        self.assertEqual(st[1],"INCOMPLETE_UTF8_NEEDS_CONTINUATION")

    def test_07_3x_causal_only_native_BPE_AFTER_incomplete_prefix(self):
        gold=[155,201,30,2]
        w3,st=native_train_weights(self.tok,gold,
                                    continuation_weight=3.)
        w1,_=native_train_weights(self.tok,gold,
                                  continuation_weight=1.)
        self.assertEqual(w3,[1.,3.,1.,1.])
        self.assertEqual(w1,[1.,1.,1.,1.])
        self.assertEqual(st[-1],"VALID_COMPLETE_UTF8")

    def test_08_extra_weight_disallowed_not_tuned(self):
        with self.assertRaisesRegex(GateFailure,"controlled q2i7"):
            native_train_weights(self.tok,[155,201,2],
                                 continuation_weight=8.)

    def test_09_native_EOS_after_incomplete_byte_never_training_target(self):
        with self.assertRaisesRegex(GateFailure,"EOS cannot follow"):
            native_train_weights(self.tok,[155,2],
                                 continuation_weight=1.)

    def test_10_invalid_utf8_before_target_is_rejected(self):
        with self.assertRaisesRegex(GateFailure,"INVALID UTF8 earlier"):
            native_train_weights(self.tok,[155,155,2],
                                 continuation_weight=3.)

    def test_11_premature_EOS_after_byte155_is_not_valid_utf8(self):
        e=sample_scored(self.tok,0,ids=[155,2])
        self.assertTrue(e["true_native_EOS_emitted"])
        self.assertTrue(e["EOS_emitted_with_INCOMPLETE_or_INVALID_UTF8"])
        self.assertEqual(e["entire_original_bytelevel_utf8_status"],
                         "INCOMPLETE_UTF8_NEEDS_CONTINUATION")
        self.assertFalse(e["strict_full_answer_exact_WITH_EOS"])

    def test_12_repeated_initial_bytes_EOS_invalid_not_just_incomplete(self):
        e=sample_scored(self.tok,0,ids=[155,155,2])
        self.assertTrue(e["EOS_emitted_with_INCOMPLETE_or_INVALID_UTF8"])
        self.assertEqual(e["entire_original_bytelevel_utf8_status"],
                         "INVALID_UTF8_SEQUENCE")

    def test_13_correct_Unicode_full_answer_with_EOS_not_byte_fragment(self):
        e=sample_scored(self.tok,0,ids=[155,201,2])
        self.assertTrue(e["strict_full_answer_exact_WITH_EOS"])
        self.assertTrue(e["true_native_EOS_emitted"])
        self.assertFalse(e["EOS_emitted_with_INCOMPLETE_or_INVALID_UTF8"])
        self.assertTrue(e["first_native_BPE_correct"])
        self.assertEqual(e["entire_original_bytelevel_utf8_status"],
                         "VALID_COMPLETE_UTF8")

    def test_14_first_native_correct_but_second_wrong_detected(self):
        e=sample_scored(self.tok,0,ids=[155,247,30,2])
        self.assertTrue(e["first_native_BPE_correct"])
        self.assertTrue(e["wrong_second_after_correct_first"])
        self.assertEqual(e["first_wrong_or_missing_native_BPE_position_1based"],2)

    def test_15_true_native_VOCAB_source_not_word_level_proxy(self):
        self.assertIn("REAL43M_GT_RUSSIAN_UTF8_1X_vs_3X",STAGE)
        self.assertEqual(SEED,119145)

    def test_16_aggregate_own_strict_AND_invalid_byte_counts(self):
        values=[sample_scored(self.tok,i) for i in range(6)]
        s=aggregate_own(values)
        self.assertEqual(s["count"],6)
        self.assertEqual(s["strict_fully_correct_WITH_native_EOS"],0)
        self.assertEqual(s["real_native_EOS_finished"],6)
        self.assertEqual(s["EOS_emitted_after_invalid_or_incomplete_utf8"],3)
        self.assertEqual(s["VALID_COMPLETE_UTF8_own_output"],3)

    def test_17_aggregate_partial_trace_denied(self):
        with self.assertRaisesRegex(GateFailure,"six DEV"):
            aggregate_own([sample_scored(self.tok,0)])

    def test_18_saved_research_3x_fp16_wrong_name_fail_closed(self):
        with self.assertRaisesRegex(GateFailure,"Q2I6 3x"):
            physical_weighted_q2i6_child(
                Path("/tmp/wrong.safetensors"),{
                    "sha256":"bad","file_bytes":0,
                },object())

    def test_19_saved_research_GT12_wrong_name_fail_closed(self):
        with self.assertRaisesRegex(GateFailure,"Q2I6 recurrent 3x"):
            physical_q2i6_gate(Path("/tmp/wrong.safetensors"),{
                "sha256":"bad","file_bytes":0,
            })


if __name__=="__main__":
    main()
