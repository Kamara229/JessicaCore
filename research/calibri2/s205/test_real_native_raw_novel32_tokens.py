"""S205 unit contracts: raw tokens only recorded; all labels only posthoc.

Synthetic fixture unit checks do NOT constitute actual CPU J7 results.
"""
from __future__ import annotations

import copy
import unittest
from calibri2.s205.real_native_raw_novel32_tokens import (
    verify_and_analyze_raw,hash_q,
)


class NativeRaw32Contracts(unittest.TestCase):
    @staticmethod
    def prepared():
        groups=["owner_A_paraphrase"]*8+["owner_B_paraphrase"]*8+[
            "reordered_owner_paraphrase"]*8+["numeric_new_wording"]*8
        group_first={"owner_A_paraphrase":5,"owner_B_paraphrase":5,
                     "reordered_owner_paraphrase":6,"numeric_new_wording":0}
        observed={name:0 for name in group_first}
        frozen,raw,flags=[],[],[]
        for i,group in enumerate(groups):
            q=f"S205 TRAIN synthetic question {i}"
            sha=hash_q(q)
            j=observed[group]
            observed[group]+=1
            first=(j<group_first[group])
            expected=[i+100,2] if group=="numeric_new_wording" else [i+100,i+300,2]
            predicted=[expected[0] if first else 1200+i,
                       *( [777+i] if group!="numeric_new_wording" else []),
                       2]
            assert predicted!=expected
            frozen.append((q,"answer text",group))
            raw.append({"question":q,"generated_ids":predicted,
                        "posthoc_expected_ids":expected})
            flags.append({"prompt_sha256":sha,"group":group,
                          "generated_tokens":len(predicted),
                          "first_token_correct":first,
                          "full_EOS_correct":False,
                          "eos_present":True})
        s190={"real_CPU_full_J7_new_TRAIN_all32":True,
              "no_gold_or_SHA_at_inference_or_routing":True,
              "hashed_per_prompt_audit":flags}
        s204={"stage":"S204_ACTUAL_NATIVE_616_CPU_J7_PROTECTED_NRA_CANDIDATE",
              "full_456_correct_EOS":456,
              "full_128_correct_EOS":128,
              "32_NEW_TRAIN_first_correct":16,
              "32_NEW_TRAIN_full_EOS_correct":0,
              "model_accepted":False}
        return raw,frozen,s190,s204

    def test_observed_native_ids_match_real_616_flags(self):
        raw,frozen,s190,s204=self.prepared()
        result=verify_and_analyze_raw(
            raw,frozen_novel=frozen,source_flags=s190,s204_full=s204,eos_id=2)
        self.assertEqual(len(result["actual_native_decoded_token_ids_per_SOURCE_TRAIN"]),32)
        self.assertEqual(result["novel32_FIRST_correct"],16)
        self.assertEqual(result["novel32_FULL_EOS_correct"],0)
        self.assertEqual(result["novel32_eos_correct"],32)
        self.assertEqual(sum(x["second_correct"] for x in result[
            "group_token_position_breakdown"].values()),8)
        self.assertFalse(result["model_accepted"])

    def test_report_flag_contradiction_rejected(self):
        raw,frozen,s190,s204=self.prepared()
        s190["hashed_per_prompt_audit"][0]["first_token_correct"]=False
        with self.assertRaises(AssertionError):
            verify_and_analyze_raw(raw,frozen_novel=frozen,source_flags=s190,
                                   s204_full=s204,eos_id=2)

    def test_duplicate_native_question_rejected(self):
        raw,frozen,s190,s204=self.prepared()
        raw[1]["question"]=raw[0]["question"]
        with self.assertRaises(AssertionError):
            verify_and_analyze_raw(raw,frozen_novel=frozen,source_flags=s190,
                                   s204_full=s204,eos_id=2)

    def test_rejection_candidate_never_accepted(self):
        raw,frozen,s190,s204=self.prepared()
        s204["model_accepted"]=True
        with self.assertRaises(AssertionError):
            verify_and_analyze_raw(raw,frozen_novel=frozen,source_flags=s190,
                                   s204_full=s204,eos_id=2)

    def test_mutation_of_actual_ids_detected(self):
        raw,frozen,s190,s204=self.prepared()
        raw[4]["generated_ids"]=[23,24,2]
        with self.assertRaises(AssertionError):
            verify_and_analyze_raw(raw,frozen_novel=frozen,source_flags=s190,
                                   s204_full=s204,eos_id=2)


if __name__ == "__main__":
    unittest.main()
