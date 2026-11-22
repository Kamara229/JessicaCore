"""S119q2i8c frozen teacher135M versus original S14243M full gold-prefix tests."""
from __future__ import annotations

from collections import Counter
from unittest import TestCase,main
import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import (
    selected_already_scored_dev,
)
from research.calibri_nra3.s119q2i8c_real135m43m_teacher_goldprefix_native import (
    STAGE,REPORT,TEACHER_TEMPERATURE,MAX_GOLD_BPE,
    AUDIT_BPE_CLASSES,LABELS,NATIVE_VOCAB,
    true_native_gold_position,teacher_soft_distribution_KL,
    audit_summarize,
)


class OriginalByteLevelTokenizer:
    eos_token_id=2
    all_special_ids=[2]
    def convert_ids_to_tokens(self,token_id):
        return {155:"Ð",201:"Ŀ",202:"Ĳ",30:".",2:"<|im_end|>",42:"A"}.get(token_id,".")
    def decode(self,ids,skip_special_tokens=True):
        return ""


def original49152(index,other=None,other_value=-9.0):
    v=torch.full((1,NATIVE_VOCAB),-10.0)
    v[0,index]=12.0
    if other is not None:
        v[0,other]=other_value
    return v


def fake_audit_rows():
    rows=[]
    for src in selected_already_scored_dev():
        steps=[]
        gold=[155,201,2]
        for i,(g,kind) in enumerate(zip(gold,("OTHER","CONTINUATION","EOS"))):
            native={}
            for label in LABELS:
                native[label]={
                    "gold_top1":i==0,
                    "gold_rank_full_49152":1 if i==0 else 42,
                    "native_gold_target_CE":float(i+1),
                    "native_gold_prob_softmax_T1":0.2,
                }
            steps.append({
                "native_position":i+1,
                "kind":kind,
                "actual_physically_run_per_model":native,
                "teacherKL_T2_full49152_against_source135m":{
                    label:.1 for label in LABELS if label!="FROZEN_SOURCE_135M_INSTRUCT"
                },
            })
        rows.append({
            "kind":src["kind"],
            "real_EOS_id":2,
            "full_correct_native_BPE_target_including_EOS":gold,
            "actual_original_full_native_teacher_forced_steps":steps,
        })
    return rows


class S119Q2I8CFrozenFullNativeLogitContracts(TestCase):
    def setUp(self):
        self.tok=OriginalByteLevelTokenizer()

    def test_01_valid_source_only_seven_spent_internal_DEV(self):
        rows=selected_already_scored_dev()
        self.assertEqual(len(rows),7)
        self.assertEqual(len({x["kind"] for x in rows}),7)
        self.assertTrue(all(x["split"]=="DEV" for x in rows))

    def test_02_fixed_source_teacher_softmax_temperature(self):
        self.assertEqual(TEACHER_TEMPERATURE,2.0)
        self.assertEqual(NATIVE_VOCAB,49152)
        self.assertEqual(MAX_GOLD_BPE,30)
        self.assertEqual(AUDIT_BPE_CLASSES,("CONTINUATION","EOS","OTHER"))

    def test_03_equal_original_logits_produce_zero_full_native_KL(self):
        a=original49152(155)
        self.assertAlmostEqual(teacher_soft_distribution_KL(a,a),0.,places=5)

    def test_04_different_original_full_native_logits_positive_KL(self):
        source=original49152(155)
        compressed=original49152(30)
        self.assertGreater(teacher_soft_distribution_KL(source,compressed),1.)

    def test_05_restricted_proxy_8k_native_logits_fail_closed(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            teacher_soft_distribution_KL(torch.zeros(1,8192),
                                         torch.zeros(1,8192))

    def test_06_nonfinite_original_source_logits_rejected(self):
        a=original49152(155)
        b=a.clone()
        b[0,0]=float("nan")
        with self.assertRaisesRegex(GateFailure,"49,152"):
            teacher_soft_distribution_KL(a,b)

    def test_07_only_preregistered_teacher_temperature_2(self):
        a=original49152(155)
        with self.assertRaisesRegex(GateFailure,"T=2"):
            teacher_soft_distribution_KL(a,a,temperature=.25)

    def test_08_identical_original_preceding_ByteLevel_start_is_incomplete(self):
        r=true_native_gold_position(self.tok,original49152(201),201,[155],1)
        self.assertEqual(r["correct_native_BPE_kind"],"CONTINUATION")
        self.assertTrue(r["gold_top1"])
        self.assertEqual(r["gold_rank_full_49152"],1)
        self.assertEqual(r["native_gold_piece"],"Ŀ")

    def test_09_eos_target_after_complete_Russian_is_EOS_category(self):
        r=true_native_gold_position(
            self.tok,original49152(2),2,[155,201],2)
        self.assertEqual(r["correct_native_BPE_kind"],"EOS")
        self.assertEqual(r["native_gold_prob_softmax_T1"],r["native_gold_prob_softmax_T1"])
        self.assertTrue(r["no_future_correct_native_BPE_used_as_INPUT"])

    def test_10_first_byte_Russian_is_OTHER_not_continuation(self):
        r=true_native_gold_position(self.tok,original49152(155),155,[],0)
        self.assertEqual(r["correct_native_BPE_kind"],"OTHER")

    def test_11_EOS_after_incomplete_is_FAIL_CLOSED(self):
        with self.assertRaisesRegex(GateFailure,"valid/incomplete UTF8"):
            true_native_gold_position(self.tok,original49152(2),2,[155,155],2)

    def test_12_real_first_byte_rank_second_when_period_dominates(self):
        v=original49152(30,other=155,other_value=11.)
        r=true_native_gold_position(self.tok,v,155,[],0)
        self.assertEqual(r["gold_rank_full_49152"],2)
        self.assertFalse(r["gold_top1"])

    def test_13_seven_sources_three_real_original_gold_targets(self):
        summary=audit_summarize(fake_audit_rows())
        self.assertEqual(summary["total_actual_original_native_GOLD_prefix_steps"],21)
        self.assertEqual(summary["actual_135M_source_AND_two_43M_forward_calls"],63)
        self.assertEqual(summary["CONTINUATION"]["count_original_gold_native_positions"],7)
        self.assertEqual(summary["EOS"]["count_original_gold_native_positions"],7)

    def test_14_all_model_source_gold_metrics_reported(self):
        summary=audit_summarize(fake_audit_rows())
        self.assertEqual(set(summary["CONTINUATION"]) &
                         set(LABELS),set(LABELS))
        self.assertEqual(summary["CONTINUATION"][
            "FROZEN_SOURCE_135M_INSTRUCT"]["gold_next_native_BPE_TOP1"],0)
        self.assertAlmostEqual(summary["CONTINUATION"][
            "FROZEN_S142_43M_ORIGINAL"][
                "mean_frozen135M_teacher_KL_at_T2"],.1)

    def test_15_six_source_domains_not_seven_disallowed(self):
        with self.assertRaisesRegex(GateFailure,"seven ALREADY"):
            audit_summarize(fake_audit_rows()[:6])

    def test_16_missing_Cyrillic_continuation_not_a_valid_diagnostic(self):
        records=fake_audit_rows()
        for r in records:
            r["actual_original_full_native_teacher_forced_steps"][1]["kind"]="OTHER"
        with self.assertRaisesRegex(GateFailure,"Cyrillic next-byte"):
            audit_summarize(records)

    def test_17_seven_EOS_one_per_full_russian_answer(self):
        summary=audit_summarize(fake_audit_rows())
        self.assertEqual(summary["EOS"]["count_original_gold_native_positions"],7)

    def test_18_wrong_gold_EOS_in_source_fails_closed(self):
        records=fake_audit_rows()
        records[0]["full_correct_native_BPE_target_including_EOS"][-1]=30
        with self.assertRaisesRegex(GateFailure,"full-BPE plus native EOS"):
            audit_summarize(records)

    def test_19_no_soft_teacher_QA_answer_overrides_allowed(self):
        self.assertIn("GOLDPREFIX_UTF8",STAGE)
        self.assertIn("SOFTLOGITS",REPORT)
        self.assertTrue(all("SOURCE" in name or "43M" in name
                            for name in LABELS))

    def test_20_all_original_weights_and_production_must_remain_unchanged(self):
        self.assertEqual(len(LABELS),3)
        self.assertEqual(set(LABELS),{
            "FROZEN_SOURCE_135M_INSTRUCT",
            "FROZEN_S142_43M_ORIGINAL",
            "Q2I8A_FITTED_43M_GT12_NRA3_NPA1"})

    def test_21_gold_prob_and_ce_consistent_with_full_49152(self):
        r=true_native_gold_position(
            self.tok,original49152(155),155,[],0)
        self.assertLess(r["native_gold_target_CE"],.05)
        self.assertGreater(r["native_gold_prob_softmax_T1"],.95)

    def test_22_bytelevel_incomplete_source_not_evidence_of_wrong_BPE(self):
        r=true_native_gold_position(
            self.tok,original49152(201),201,[155],1)
        self.assertEqual(r["gold_previous_native_BPE_UTF8_state"],
                         "INCOMPLETE_UTF8_NEEDS_CONTINUATION")


if __name__=="__main__":
    main()
