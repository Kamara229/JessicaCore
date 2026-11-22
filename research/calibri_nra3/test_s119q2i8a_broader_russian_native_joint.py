"""S119q2i8a fail-closed greater source diversity and 2-epoch A/B contracts."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest import TestCase,main
import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import (
    STAGE as SOURCE_STAGE,CASES,ORIGIN,KINDS,
    rows as fresh_rows,source_sha,verify as verify_source,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_rows,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    native_train_weights,
)
from research.calibri_nra3.s119q2i8a_real43m_broader_russian_joint import (
    STAGE,NEW_TRAIN,NEW_DEV,OLD_DEV,OLD_TRAIN_REHEARSAL,
    TOTAL_TRAIN,EPOCHS,MAX_GENERATED_NATIVE,NEURAL_PARAMS,
    VARIANTS,CONTINUATION_WEIGHT,SEED,
    verify_physical_1x_parent,verify_physical_1x_gt,
    prepare_native_original,score,
)


class NativeMinimal:
    eos_token_id=2
    all_special_ids=[2]
    def convert_ids_to_tokens(self,x):
        return {155:"Ð",201:"Ĳ",2:"<|im_end|>",30:"."}.get(x,".")
    def decode(self,ids,skip_special_tokens=True):
        return "placeholder"


def fake_scored(n):
    return [{
        "true_native_full_EOS_correct":(i==0),
        "text_exact_even_if_not_finished":(i==0),
        "native_EOS":True,
        "real_EOS_but_UNFINISHED_or_INVALID_UTF8":i%2==0,
        "valid_full_UTF8_bytes":i%2==1,
        "first_gold_BPE_correct":i%3==0,
        "first_native_correct_but_wrong_second":i%3==0 and i!=0,
        "original_full_decoder_calls":3,
        "domain":"unicode_word_copy",
    } for i in range(n)]


class S119Q2I8ABroaderRussianSourceContracts(TestCase):
    def test_01_source_seven_domains_70_14(self):
        m=verify_source()
        self.assertEqual(len(KINDS),7)
        self.assertEqual(m["TRAIN"],70)
        self.assertEqual(m["DEV"],14)
        self.assertEqual(m["stage"],SOURCE_STAGE)
        self.assertTrue(m["no_overlap_prior_or_spent_Q2G_I1_I6"])

    def test_02_exact_ten_TRAIN_per_each_domain(self):
        self.assertEqual(Counter(x["kind"] for x in fresh_rows("TRAIN")),
                         {k:10 for k in KINDS})

    def test_03_exact_two_INTERNAL_DEV_per_domain(self):
        self.assertEqual(Counter(x["kind"] for x in fresh_rows("DEV")),
                         {k:2 for k in KINDS})

    def test_04_no_internal_DEV_prompt_in_training(self):
        self.assertFalse(
            {x["prompt"] for x in fresh_rows("TRAIN")} &
            {x["prompt"] for x in fresh_rows("DEV")})
        self.assertEqual(len({x["prompt"] for x in
                              fresh_rows("TRAIN")+fresh_rows("DEV")}),84)

    def test_05_all_original_synthetic_not_scraped_natural(self):
        m=verify_source()
        self.assertTrue(m["original_synthetic_created_for_experiment_NOT_licensed_natural"])
        self.assertTrue(m["DEV_internal_visible_small_NOT_external_independent_blind"])
        self.assertTrue(all(x["origin"]==ORIGIN for x in
                            fresh_rows("TRAIN")+fresh_rows("DEV")))

    def test_06_no_sealed_or_fake_independent_EVAL_in_training_API(self):
        for split in ("EVAL","SEALED","S118_GOLD",None):
            with self.assertRaisesRegex(ValueError,"registered original"):
                fresh_rows(split)

    def test_07_stable_sha_of_exact_preregistered_source_splits(self):
        p=verify_source()
        self.assertEqual(p["TRAIN_SHA256"],source_sha(fresh_rows("TRAIN")))
        self.assertEqual(p["DEV_SHA256"],source_sha(fresh_rows("DEV")))
        self.assertEqual(len(p["TRAIN_SHA256"]),64)
        self.assertNotEqual(p["TRAIN_SHA256"],p["DEV_SHA256"])

    def test_08_only_existing_old_six_SOURCE_TRAIN_rehearsal(self):
        a=[x for x in old_rows("TRAIN")
           if x["id"].rsplit("_",1)[-1]=="00"]
        b=[x for x in old_rows("DEV")
           if x["id"].rsplit("_",1)[-1]=="06"]
        self.assertEqual(len(a),OLD_TRAIN_REHEARSAL)
        self.assertEqual(len(b),OLD_DEV)
        self.assertFalse({x["prompt"] for x in a}&
                         {x["prompt"] for x in b})

    def test_09_all_new_source_completions_are_short_FULL_answers(self):
        self.assertTrue(all(
            1<=len(x["completion"])<=25 and x["completion"].strip()
            for x in fresh_rows("TRAIN")+fresh_rows("DEV")))

    def test_10_arithmetic_answer_grounded_by_written_prompt(self):
        cases=dict(CASES["numeric_arithmetic"])
        self.assertIn(("В корзине 7 слив, добавили 4. Сколько слив теперь? Только число.","11"),
                      cases.items())
        self.assertIn(("Из 27 фишек потеряли 8. Сколько фишек осталось?","19"),
                      cases.items())

    def test_11_training_budget_is_larger_but_matched(self):
        self.assertEqual((NEW_TRAIN,OLD_TRAIN_REHEARSAL,TOTAL_TRAIN),
                         (70,6,76))
        self.assertEqual(NEURAL_PARAMS,368051)
        self.assertEqual(EPOCHS,2)
        self.assertEqual(TOTAL_TRAIN*EPOCHS,152)

    def test_12_max_generated_tokens_are_genuine_NATIVE_BPE(self):
        self.assertEqual(MAX_GENERATED_NATIVE,18)
        self.assertLess(MAX_GENERATED_NATIVE,24)
        self.assertEqual(NEW_DEV,14)
        self.assertEqual(OLD_DEV,6)

    def test_13_same_recurrent_vs_local_architecture_controls(self):
        self.assertEqual(VARIANTS,("GT_RECURRENT","GT_NOCARRY_LOCAL"))
        self.assertEqual(CONTINUATION_WEIGHT,1.0)
        self.assertEqual(SEED,119147)

    def test_14_research_only_checkpoint_wrong_name_DENIED(self):
        with self.assertRaisesRegex(GateFailure,"q2i7 1x entire"):
            verify_physical_1x_parent(
                Path("/tmp/fake_wrong.safetensors"),
                {"sha256":"x","file_bytes":0},object())

    def test_15_research_only_GT12_wrong_name_DENIED(self):
        with self.assertRaisesRegex(GateFailure,"GT12 separate"):
            verify_physical_1x_gt(
                Path("/tmp/fake_wrong.safetensors"),
                {"sha256":"x","file_bytes":0})

    def test_16_unauthorized_external_source_denied(self):
        row=dict(fresh_rows("TRAIN")[0])
        row["origin"]="unverified_natural_web_corpus"
        with self.assertRaisesRegex(GateFailure,"registered 70TRAIN"):
            prepare_native_original(NativeMinimal(),row)

    def test_17_scored_new_14_dev_full_accuracy_and_EOS_separate(self):
        scores=score(fake_scored(14))
        self.assertEqual(scores["source_count"],14)
        self.assertEqual(scores["correct_full_native_ANSWER_and_EOS"],1)
        self.assertEqual(scores["true_native_EOS_emitted"],14)
        self.assertEqual(scores["native_EOS_after_incomplete_or_INVALID_UTF8"],7)

    def test_18_old_dev_retention_is_not_training(self):
        scores=score(fake_scored(6))
        self.assertEqual(scores["source_count"],6)
        self.assertEqual(scores["original_43M_Core12_native_decoder_forwards"],18)
        self.assertEqual(scores["full_UTF8_valid_own_native"],3)

    def test_19_wrong_scored_source_count_DENIED(self):
        with self.assertRaisesRegex(GateFailure,"14 NEWDEV or 6 old-source"):
            score(fake_scored(12))

    def test_20_regular_1x_native_fullanswer_EOS_uniform_weights(self):
        tok=NativeMinimal()
        weights,states=native_train_weights(
            tok,[155,201,2],continuation_weight=1.0)
        self.assertEqual(weights,[1.,1.,1.])
        self.assertEqual(states[1],"INCOMPLETE_UTF8_NEEDS_CONTINUATION")

    def test_21_older_default_3x_still_backward_compatible(self):
        tok=NativeMinimal()
        weights,states=native_train_weights(tok,[155,201,2])
        self.assertEqual(weights,[1.,3.,1.])
        self.assertEqual(len(states),3)

    def test_22_no_source_only_metric_mistaken_for_BLIND_language_quality(self):
        self.assertIn("BROADER_RU_NATIVE_70TRAIN_14DEV",STAGE)
        self.assertTrue(verify_source()["DEV_internal_visible_small_NOT_external_independent_blind"])


if __name__=="__main__":
    main()
