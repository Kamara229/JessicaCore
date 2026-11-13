"""S205 unit-only novel TRAIN transcript logic: not model inference."""
from __future__ import annotations
import copy
import unittest

from calibri2.s205.novel32_native_posthoc import diagnose,STAGE


class S205NativeTraceReasoning(unittest.TestCase):
    @staticmethod
    def fixtures():
        groups={"owner_A_paraphrase":5,"owner_B_paraphrase":5,
                "reordered_owner_paraphrase":6,"numeric_new_wording":0}
        rows=[]
        grouped={}
        i=1
        for group,first_n in groups.items():
            grouped[group]={"count":8,"first_correct":first_n,
                            "full_EOS_correct":0,"eos_present":8}
            for j in range(8):
                n=(3 if group!="numeric_new_wording" else
                   4 if j==0 else 2 if j==1 else 3)
                rows.append({"prompt_sha256":f"{i:064x}","group":group,
                             "generated_tokens":n,"first_token_correct":j<first_n,
                             "full_EOS_correct":False,"eos_present":True})
                i+=1
        s190={"real_CPU_full_J7_new_TRAIN_all32":True,
              "no_gold_or_SHA_at_inference_or_routing":True,
              "original_real_CPU_FULL_EOS_456":456,
              "new128_real_CPU_FULL_EOS":128,
              "hashed_per_prompt_audit":rows,
              "NEW_TRAIN_group_metrics":grouped,
              "NEW_TRAIN_first_correct":16,
              "NEW_TRAIN_full_EOS_correct":0,
              "NEW_TRAIN_eos_present":32}
        s204={"stage":"S204_ACTUAL_NATIVE_616_CPU_J7_PROTECTED_NRA_CANDIDATE",
              "source_S197_pinned_GitHub_run":43369398297,
              "full_456_correct_EOS":456,
              "full_128_correct_EOS":128,
              "model_accepted":False,
              "S204_native_CPU_456_128_source_hook":{
                  "real_CPU_inference_completed_before_gold_scoring":True,
                  "S73_actual_generated_ids":{f"s73-{i}":[1,2,3] for i in range(4)}
              },
              "32_NEW_TRAIN_first_correct":16,
              "32_NEW_TRAIN_full_EOS_correct":0,
              "CPU_ratio_independently_paired_for_new_candidate":False,
              "total_physical_budget":{
                  "all_in_bits":30612352.868287183,
                  "conservative_bpp_excluding_zero_structural_growth":0.7072496380455623,
              }}
        return s204,s190

    def test_full_32_posthoc_not_neural_inference(self):
        s204,s190=self.fixtures()
        r=diagnose(s204,s190)
        self.assertEqual(r["stage"],STAGE)
        self.assertFalse(r["retrain_or_decoder_inference_ran"])
        self.assertEqual(r["counts"]["first_correct"],16)
        self.assertEqual(r["counts"]["full_EOS_correct"],0)
        self.assertEqual(r["counts"]["EOS_present"],32)
        self.assertEqual(r["counts"]["provable_second_token_failures"],16)
        self.assertEqual(r["numeric_first_wrong_count"],8)
        self.assertEqual(r["budget_bytes_over_hard_limit"],39224)
        self.assertTrue(r["independent_candidate_vs_parent_CPU_speed_STILL_UNVERIFIED"])
        self.assertFalse(r["training_candidate_accepted"])

    def test_no_fabricated_or_duplicate_native_sha(self):
        s204,s190=self.fixtures()
        s190["hashed_per_prompt_audit"][1]["prompt_sha256"]=s190[
            "hashed_per_prompt_audit"][0]["prompt_sha256"]
        with self.assertRaises(AssertionError):
            diagnose(s204,s190)

    def test_report_crosscheck_failure(self):
        s204,s190=self.fixtures()
        s204["32_NEW_TRAIN_first_correct"]=17
        with self.assertRaises(AssertionError):
            diagnose(s204,s190)

    def test_fail_on_not_native_or_wrong_denominator(self):
        s204,s190=self.fixtures()
        s190["real_CPU_full_J7_new_TRAIN_all32"]=False
        with self.assertRaises(AssertionError):
            diagnose(s204,s190)
        s204,s190=self.fixtures()
        s204["total_physical_budget"]["all_in_bits"]=1.0
        with self.assertRaises(AssertionError):
            diagnose(s204,s190)


if __name__=="__main__":
    unittest.main()
