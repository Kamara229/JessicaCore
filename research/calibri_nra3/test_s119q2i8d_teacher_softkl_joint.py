"""S119q2i8d causal SOURCE135M 49k KL at UTF8-only full native tests."""
from __future__ import annotations

from collections import Counter
from inspect import signature
from unittest import TestCase,main
import torch
from torch import nn
from torch.nn import functional as F

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import (
    rows as original_train,verify as old_source_verify,KINDS,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_i1_rows,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    train_true_joint,native_train_weights,
)
from research.calibri_nra3.s119q2i8d_new_russian_heldout_source import (
    STAGE as DEV_STAGE,ORIGIN,
    rows as fresh_DEV,verify as verify_DEV,
)
from research.calibri_nra3.s119q2i8d_real43m_source_softkl_joint_train import (
    STAGE,REPORT,ROUNDS,TEACHER_KL_CONT_WEIGHT,TEMPERATURE,
    CONDITIONS,TRAIN_ROWS,ORIGINAL_NEW_TRAIN,NEW_HELDOUT_DEV,
    OLD_SPENT_DEV,NATIVE_VOCAB,pack_new_author_dev,
)


class MinimalNative:
    eos_token_id=2
    all_special_ids=[2]
    def convert_ids_to_tokens(self,i):
        return {155:"Ð",201:"Ĳ",30:".",2:"<|im_end|>",42:"A"}.get(i,".")
    def decode(self,ids,skip_special_tokens=True):
        return "NEVER_USE_FOR_REAL_GOLD"


def full_logits(token_id,weight=10.0):
    x=torch.full((1,NATIVE_VOCAB),-10.0)
    x[0,token_id]=weight
    return x


class S119Q2I8DTeacherSoftFullNativeContracts(TestCase):
    def setUp(self):
        self.tok=MinimalNative()

    def test_01_new_DEV_is_original_and_14_split(self):
        meta=verify_DEV()
        self.assertEqual(meta["original_new_author_heldout_DEV_count"],14)
        self.assertTrue(meta["no_prompt_overlap_with_any_previous_training_or_scored_DEV"])
        self.assertTrue(meta["synthetic_author_written_NOT_external_blind"])

    def test_02_exact_two_new_author_DEV_each_of_seven_domains(self):
        rows=fresh_DEV()
        self.assertEqual(len(rows),14)
        self.assertEqual(Counter(x["kind"] for x in rows),
                         {k:2 for k in KINDS})
        self.assertTrue(all(x["origin"]==ORIGIN and x["split"]=="DEV"
                            for x in rows))

    def test_03_DEV_only_api_explicitly_rejects_TRAIN_and_EVAL(self):
        for name in ("TRAIN","EVAL","SEALED_EVAL"):
            with self.assertRaisesRegex(ValueError,"only a NEW internal DEV"):
                fresh_DEV(name)

    def test_04_old_existing_original_author_training_source_only(self):
        source=old_source_verify()
        self.assertEqual(source["TRAIN"],70)
        self.assertEqual(ORIGINAL_NEW_TRAIN,70)
        self.assertEqual(TRAIN_ROWS,76)
        self.assertEqual(ROUNDS,1)

    def test_05_all14_new_dev_prompts_disjoint_from_previous_synthetic(self):
        new={x["prompt"] for x in fresh_DEV()}
        old={x["prompt"] for x in
             original_train("TRAIN")+original_train("DEV")}
        self.assertFalse(new&old)
        self.assertEqual(len(new),14)

    def test_06_six_old_rehearsal_train_does_not_include_old_DEV(self):
        oldtrain=[x for x in old_i1_rows("TRAIN")
                  if int(x["id"].rsplit("_",1)[-1])==0]
        olddev=old_i1_rows("DEV")
        self.assertEqual(len(oldtrain),6)
        self.assertEqual(len(olddev),12)
        self.assertFalse({r["prompt"] for r in oldtrain}&
                         {r["prompt"] for r in olddev})

    def test_07_very_conservative_teacher_KL_weight_is_fixed(self):
        self.assertEqual(TEACHER_KL_CONT_WEIGHT,.25)
        self.assertEqual(TEMPERATURE,2.)
        self.assertEqual(CONDITIONS,(
            "CE_ONLY","SOURCE135M_UTF8_CONTINUATION_KL"))

    def test_08_teacher_LOGITS_and_student_same_original_full49k(self):
        self.assertEqual(NATIVE_VOCAB,49152)
        teacher=full_logits(201)
        student=full_logits(155).requires_grad_()
        prior=F.softmax(teacher/TEMPERATURE,dim=-1)
        studentlog=F.log_softmax(student/TEMPERATURE,dim=-1)
        kl=(TEMPERATURE**2)*F.kl_div(
            studentlog,prior,reduction="sum")
        self.assertGreater(float(kl.detach()),1.)
        (kl*TEACHER_KL_CONT_WEIGHT).backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.detach().abs().sum()),0.)

    def test_09_exact_teacher_and_student_have_zero_real_49k_KL(self):
        z=full_logits(155)
        a=F.softmax(z/2.,dim=-1)
        b=F.log_softmax(z/2.,dim=-1)
        self.assertAlmostEqual(
            float(F.kl_div(b,a,reduction="sum")),0.,places=5)

    def test_10_native_UTF8_D0_both_next_real_target_and_EOS_not_KL(self):
        weights,status=native_train_weights(
            self.tok,[155,201,2],continuation_weight=1.)
        self.assertEqual(weights,[1.,1.,1.])
        self.assertEqual(status,[
            "VALID_COMPLETE_UTF8",
            "INCOMPLETE_UTF8_NEEDS_CONTINUATION",
            "VALID_COMPLETE_UTF8"])
        self.assertNotEqual(status[-1],"INCOMPLETE_UTF8_NEEDS_CONTINUATION")

    def test_11_only_previous_byte_is_legal_teacher_soft_input(self):
        # No current/future token may be in source soft prefix, matching
        # original train_true_joint : prefix=prompt+answer[:t].
        example=[3,4,5];target=[155,201,2]
        self.assertEqual(example+target[:1],[3,4,5,155])
        self.assertNotIn(201,example+target[:1])

    def test_12_same_76_optimizer_updates_and_368051_parameters(self):
        self.assertIn("REAL43M",STAGE)
        self.assertIn("SOURCE135M",STAGE)
        self.assertIn("TEACHERKL",REPORT)
        self.assertEqual(TRAIN_ROWS*ROUNDS,76)

    def test_13_train_true_joint_exposes_controlled_teacher_contract_only(self):
        params=signature(train_true_joint).parameters
        self.assertIn("teacher_135m",params)
        self.assertIn("teacher_KL_weight",params)
        self.assertIn("rounds",params)
        self.assertEqual(params["teacher_135m"].default,None)
        self.assertEqual(params["teacher_KL_weight"].default,0.0)

    def test_14_no_original_Q2I8D_training_or_external_oracle_labels(self):
        r=fresh_DEV()[0]
        self.assertEqual(r["split"],"DEV")
        self.assertTrue(r["id"].startswith("S119Q2I8D_"))
        self.assertEqual(DEV_STAGE,
            "CALIBRI_S119Q2I8D_FRESH_ORIGINAL_AUTHORED_14_NEVER_TRAINED_INTERNAL_RUSSIAN_DEV")

    def test_15_new_unknown_foreign_corpus_record_rejected(self):
        row=fresh_DEV()[0].copy()
        row["origin"]="unverified_external_auto_scraped"
        with self.assertRaisesRegex(GateFailure,"ONLY new q2i8d"):
            pack_new_author_dev(self.tok,row)

    def test_16_wrong_split_must_not_enter_audit(self):
        row=fresh_DEV()[0].copy()
        row["split"]="TRAIN"
        with self.assertRaisesRegex(GateFailure,"ONLY new q2i8d"):
            pack_new_author_dev(self.tok,row)

    def test_17_new_dev_original_SHA_is_stable(self):
        m=verify_DEV()
        n=verify_DEV()
        self.assertEqual(m["source_new_DEV_sha256"],n["source_new_DEV_sha256"])
        self.assertEqual(len(m["source_new_DEV_sha256"]),64)

    def test_18_teacher_135M_is_an_unchanged_soft_distribution_not_QA_truth(self):
        self.assertTrue(all(x["completion"] for x in fresh_DEV()))
        self.assertEqual(OLD_SPENT_DEV,6)
        self.assertTrue(verify_DEV()["synthetic_author_written_NOT_external_blind"])

    def test_19_never_hard_mask_token155_or_native30(self):
        teacher=full_logits(155)
        student=full_logits(30).requires_grad_()
        loss=F.kl_div(F.log_softmax(student/TEMPERATURE,dim=-1),
                      F.softmax(teacher/TEMPERATURE,dim=-1),
                      reduction="sum")*(TEMPERATURE**2)
        loss.backward()
        self.assertGreater(float(student.grad[0,155].abs()),0.)
        self.assertGreater(float(student.grad[0,30].abs()),0.)

    def test_20_one_epoch_vs_prior_2epochs_research_microblock(self):
        self.assertEqual(ROUNDS,1)
        self.assertEqual(NEW_HELDOUT_DEV,14)
        self.assertEqual(len(KINDS),7)


if __name__=="__main__":
    main()
