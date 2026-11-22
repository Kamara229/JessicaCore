"""S119q2i8c original 135M soft full-native teacher diagnostic tests.

All numerical source/student divergence is calculated from complete
49152-class physically evaluated logits. Teacher is never a QA oracle.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest import TestCase,main

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i8c_real135m43m_teacher_goldprefix_native import (
    STAGE,REPORT,TEACHER_TEMPERATURE,MAX_GOLD_BPE,
    AUDIT_BPE_CLASSES,LABELS,NATIVE_VOCAB,KIND_COUNT,
    selected_already_scored_dev,teacher_soft_distribution_KL,
    true_native_gold_position,audit_summarize,
)


class SimNative:
    eos_token_id=2
    all_special_ids=[2]
    def convert_ids_to_tokens(self,x):
        return {155:"Ð",201:"Ĳ",30:".",2:"<|im_end|>",
                90:"A"}.get(x,str(x))


def logits(winner,other=None):
    s=torch.full((1,NATIVE_VOCAB),-20.0)
    s[0,winner]=10.0
    if other is not None:
        s[0,other]=9.0
    return s


def full7():
    from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import KINDS
    entries=[]
    for kind in KINDS:
        rows=[]
        for idx,(t,target,prefix) in enumerate([
            ("OTHER",155,[]),
            ("CONTINUATION",201,[155]),
            ("EOS",2,[155,201]),
        ]):
            masks={label:{
                "native_step_index_1based":idx+1,
                "native_gold_target_id":target,
                "gold_top1":idx==2,
                "gold_rank_full_49152":idx+1,
                "native_gold_target_CE":float(idx+1),
                "native_gold_prob_softmax_T1":0.25,
                "correct_native_BPE_kind":t,
            } for label in LABELS}
            rows.append({
                "kind":t,
                "actual_physically_run_per_model":masks,
                "teacherKL_T2_full49152_against_source135m":{
                    lab:0.5 for lab in LABELS if
                    lab!="FROZEN_SOURCE_135M_INSTRUCT"
                },
            })
        entries.append({
            "kind":kind,"real_EOS_id":2,
            "full_correct_native_BPE_target_including_EOS":[155,201,2],
            "actual_original_full_native_teacher_forced_steps":rows,
        })
    return entries


class TestS119Q2I8CSourceAndCompressedNativeGoldPrefix(TestCase):
    def setUp(self):
        self.tok=SimNative()

    def test_01_teacher_is_pinned_existing_source_not_new_oracle(self):
        self.assertIn("REAL135M_TEACHER_VS_43M",STAGE)
        self.assertEqual(TEACHER_TEMPERATURE,2.0)
        self.assertEqual(NATIVE_VOCAB,49152)
        self.assertEqual(MAX_GOLD_BPE,30)
        self.assertTrue(REPORT.endswith(".json"))

    def test_02_seven_previously_scored_dev_from_original_source(self):
        dev=selected_already_scored_dev()
        self.assertEqual(len(dev),KIND_COUNT)
        self.assertEqual(len({x["kind"] for x in dev}),7)
        self.assertTrue(all(x["split"]=="DEV" for x in dev))
        self.assertTrue(all(x["id"].endswith("_10") for x in dev))

    def test_03_only_same_three_actual_original_neural_variants(self):
        self.assertEqual(LABELS,(
            "FROZEN_SOURCE_135M_INSTRUCT",
            "FROZEN_S142_43M_ORIGINAL",
            "Q2I8A_FITTED_43M_GT12_NRA3_NPA1",
        ))

    def test_04_full_native_source_self_KL_zero(self):
        source=logits(155,201)
        self.assertLess(teacher_soft_distribution_KL(source,source),0.00002)

    def test_05_teacher_KL_student_wrong_ranking_positive(self):
        teacher=logits(201,155)
        compressed=logits(155,201)
        self.assertGreater(teacher_soft_distribution_KL(
            teacher,compressed),0.01)

    def test_06_teacher_KL_prohibits_sparse_proxy_vocab(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            teacher_soft_distribution_KL(torch.ones(1,20),torch.ones(1,20))

    def test_07_teacher_KL_rejects_different_shapes(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            teacher_soft_distribution_KL(
                torch.ones(1,NATIVE_VOCAB),torch.ones(NATIVE_VOCAB))

    def test_08_teacher_KL_denies_adaptive_DEV_temp_hyperparameter(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            teacher_soft_distribution_KL(
                torch.ones(1,NATIVE_VOCAB),torch.ones(1,NATIVE_VOCAB),
                temperature=4.)

    def test_09_teacher_KL_NAN_logits_fail_closed(self):
        bad=logits(155);bad[0,155]=float("nan")
        with self.assertRaisesRegex(GateFailure,"49,152"):
            teacher_soft_distribution_KL(bad,logits(155))

    def test_10_gold_first_Cyrillic_byte_can_be_correct(self):
        a=true_native_gold_position(self.tok,logits(155),155,[],0)
        self.assertEqual(a["correct_native_BPE_kind"],"OTHER")
        self.assertTrue(a["gold_top1"])
        self.assertEqual(a["gold_rank_full_49152"],1)
        self.assertEqual(a["native_gold_piece"],"Ð")

    def test_11_gold_SECOND_byte_after_D0_is_UTF8_CONTINUATION(self):
        a=true_native_gold_position(
            self.tok,logits(155,201),201,[155],1)
        self.assertEqual(a["correct_native_BPE_kind"],"CONTINUATION")
        self.assertEqual(a["gold_rank_full_49152"],2)
        self.assertFalse(a["gold_top1"])
        self.assertEqual(a["native_gold_piece"],"Ĳ")

    def test_12_gold_native_EOS_scored_only_after_complete_byte(self):
        a=true_native_gold_position(
            self.tok,logits(2),2,[155,201],2)
        self.assertEqual(a["correct_native_BPE_kind"],"EOS")
        self.assertEqual(a["gold_rank_full_49152"],1)

    def test_13_missing_second_byte_before_native_EOS_is_not_valid_target(self):
        # A correct gold target cannot terminate within D0 incomplete.
        with self.assertRaisesRegex(GateFailure,"known CORRECT"):
            true_native_gold_position(self.tok,logits(2),2,[155,155],2)

    def test_14_actual_full_49152_native_rank_required(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            true_native_gold_position(self.tok,torch.zeros(1,30),201,[155],1)

    def test_15_three_groups_source_train_diagnostic_no_optimizer(self):
        groups=audit_summarize(full7())
        self.assertEqual(groups["CONTINUATION"]["count_original_gold_native_positions"],7)
        self.assertEqual(groups["EOS"]["count_original_gold_native_positions"],7)
        self.assertEqual(groups["OTHER"]["count_original_gold_native_positions"],7)
        self.assertEqual(groups["total_actual_original_native_GOLD_prefix_steps"],21)
        self.assertEqual(groups["actual_135M_source_AND_two_43M_forward_calls"],63)

    def test_16_same_native_groups_teacher_KL_averages(self):
        groups=audit_summarize(full7())
        data=groups["CONTINUATION"]["FROZEN_S142_43M_ORIGINAL"]
        self.assertEqual(data["gold_next_native_BPE_TOP1"],0)
        self.assertAlmostEqual(
            data["mean_frozen135M_teacher_KL_at_T2"],.5)
        self.assertEqual(data["mean_gold_next_native_BPE_rank"],2)

    def test_17_missing_old_dev_domain_rejected(self):
        with self.assertRaisesRegex(GateFailure,"seven ALREADY SCORED"):
            audit_summarize(full7()[:-1])

    def test_18_inconsistent_native_EOS_gold_target_rejected(self):
        data=full7()
        data[0]["full_correct_native_BPE_target_including_EOS"]=[155,201,201]
        with self.assertRaisesRegex(GateFailure,"full-BPE"):
            audit_summarize(data)

    def test_19_unseen_wrong_tensor_native_class_fails_closed(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            true_native_gold_position(
                self.tok,torch.zeros(1,NATIVE_VOCAB),NATIVE_VOCAB,[],0)

    def test_20_zero_training_or_source_as_gold_decision(self):
        self.assertEqual(AUDIT_BPE_CLASSES,("CONTINUATION","EOS","OTHER"))
        self.assertEqual(KIND_COUNT,7)


if __name__=="__main__":
    main()
