"""S119q2i6 18-new+6-rehearsal UTF8 true joint training controls."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from unittest import TestCase,main

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_i1_rows,
)
from research.calibri_nra3.s119q2i6_fresh_ru_utf8_curriculum import (
    KINDS,CASES,AUTHOR_ORIGIN,STAGE as SOURCE_STAGE,
    rows as new_ru_rows,verify as verify_source,
    source_sha,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    STAGE,ROUNDS,NEW_TRAIN,REHEARSAL_TRAIN,TOTAL_TRAIN,
    NEW_DEV,OLD_DEV,GAIN,CONTINUATION_WEIGHT,
    CONDITIONS,GATES,TOTAL_EXPECTED_TRAINABLE,
    native_train_weights,prepare_new,
)


class MinimalNativeBytelevel:
    eos_token_id=2
    all_special_ids=[2]
    def convert_ids_to_tokens(self,id):
        return {155:"Ð",201:"Ĳ",202:"ĳ",30:".",
                2:"<|im_end|>",40:"("}.get(id,".")
    def decode(self,ids,skip_special_tokens=True):
        raise RuntimeError("not needed for native byte weights")


class S119Q2I6TrueUtf8JointContracts(TestCase):
    def test_01_new_original_train_and_dev_sizes(self):
        meta=verify_source()
        self.assertEqual(meta["fresh_original_author_TRAIN"],18)
        self.assertEqual(meta["fresh_original_author_DEV"],6)
        self.assertTrue(meta["no_overlap_any_previous_source_or_spent_evaluation"])
        self.assertTrue(meta["DEV_is_small_inspectable_NOT_independent_BLIND"])

    def test_02_new_train_balanced_three_each_six_categories(self):
        a=new_ru_rows("TRAIN")
        self.assertEqual(len(a),NEW_TRAIN)
        self.assertEqual(Counter(x["kind"] for x in a),{k:3 for k in KINDS})
        self.assertTrue(all(x["origin"]==AUTHOR_ORIGIN for x in a))

    def test_03_new_DEV_one_each_domain_not_optimizer(self):
        a=new_ru_rows("DEV")
        self.assertEqual(len(a),NEW_DEV)
        self.assertEqual(Counter(x["kind"] for x in a),{k:1 for k in KINDS})
        self.assertTrue(all(x["split"]=="DEV" for x in a))
        self.assertFalse({x["prompt"] for x in a}&
                         {x["prompt"] for x in new_ru_rows("TRAIN")})

    def test_04_original_source_sha_exact(self):
        p=verify_source()
        self.assertEqual(p["fresh_TRAIN_SHA256"],source_sha(new_ru_rows("TRAIN")))
        self.assertEqual(p["fresh_DEV_SHA256"],source_sha(new_ru_rows("DEV")))
        self.assertEqual(len(p["fresh_TRAIN_SHA256"]),64)

    def test_05_no_reserved_EVAL_exposure(self):
        with self.assertRaisesRegex(ValueError,"Only new original-author"):
            new_ru_rows("EVAL")
        with self.assertRaisesRegex(ValueError,"Only new original-author"):
            new_ru_rows("SEALED_EVAL")

    def test_06_old_six_rehearsal_only_FROM_TRAIN_split(self):
        old=[r for r in old_i1_rows("TRAIN")
             if int(r["id"].split("_")[-1])==0]
        old_dev=old_i1_rows("DEV")
        self.assertEqual(len(old),REHEARSAL_TRAIN)
        self.assertEqual(len(old_dev),12)
        self.assertFalse({r["prompt"] for r in old}&
                         {r["prompt"] for r in old_dev})
        self.assertEqual(len(old)+len(new_ru_rows("TRAIN")),TOTAL_TRAIN)

    def test_07_real_gpt_bytelevel_initial_D0_requires_continuation_weight(self):
        tok=MinimalNativeBytelevel()
        w,status=native_train_weights(tok,[155,201,2])
        self.assertEqual(w,[1.,3.,1.])
        self.assertEqual(status,[
            "VALID_COMPLETE_UTF8","INCOMPLETE_UTF8_NEEDS_CONTINUATION",
            "VALID_COMPLETE_UTF8"])

    def test_08_real_gpt_bytelevel_complete_ascii_EOS_normal_weight(self):
        tok=MinimalNativeBytelevel()
        w,status=native_train_weights(tok,[30,2])
        self.assertEqual(w,[1.,1.])
        self.assertEqual(status,["VALID_COMPLETE_UTF8","VALID_COMPLETE_UTF8"])

    def test_09_gpt_bytelevel_two_real_Russian_characters_weight_each(self):
        tok=MinimalNativeBytelevel()
        w,status=native_train_weights(tok,[155,201,155,202,2])
        self.assertEqual(w,[1.,3.,1.,3.,1.])
        self.assertEqual(status.count("INCOMPLETE_UTF8_NEEDS_CONTINUATION"),2)

    def test_10_wrong_second_initial_UTF8_byte_disallowed_in_TRAIN(self):
        tok=MinimalNativeBytelevel()
        with self.assertRaisesRegex(GateFailure,"INVALID UTF8 earlier prefix"):
            native_train_weights(tok,[155,155,2])

    def test_11_partial_Russian_native_character_before_EOS_fail_closed(self):
        tok=MinimalNativeBytelevel()
        with self.assertRaisesRegex(GateFailure,"EOS cannot follow"):
            native_train_weights(tok,[155,2])

    def test_12_no_terminal_real_EOS_fails_closed(self):
        tok=MinimalNativeBytelevel()
        with self.assertRaisesRegex(GateFailure,"terminal EOS"):
            native_train_weights(tok,[155,201,30])

    def test_13_original_native_49k_trainable_budget_matched_GT(self):
        self.assertEqual(TOTAL_EXPECTED_TRAINABLE,368051)
        self.assertEqual(GATES,12)
        self.assertEqual(ROUNDS,3)
        self.assertEqual(ROUNDS*TOTAL_TRAIN,72)
        self.assertEqual(CONDITIONS,("GT_RECURRENT","GT_NOCARRY_LOCAL"))

    def test_14_no_hardcoded_Russian_native_tokens_or_masks(self):
        self.assertEqual(CONTINUATION_WEIGHT,3.0)
        self.assertEqual(GAIN,.25)
        self.assertTrue(all(len(v)==4 for v in CASES.values()))
        self.assertFalse(any("�" in r["completion"]
                             for r in new_ru_rows("TRAIN")))

    def test_15_source_no_fake_independent_quality_certification(self):
        self.assertIn("UTF8_MULTIBPE_CONTINUATION_TRAIN",STAGE)
        self.assertIn("RUSSIAN_UTF8_MULTIBPE",SOURCE_STAGE)
        self.assertEqual(OLD_DEV,6)

    def test_16_external_unapproved_source_row_rejected(self):
        from unittest.mock import patch
        src=deepcopy(new_ru_rows("TRAIN")[0])
        src["origin"]="unknown_external_source"
        with self.assertRaisesRegex(GateFailure,"pristine"):
            prepare_new(MinimalNativeBytelevel(),src)


if __name__=="__main__":
    main()
