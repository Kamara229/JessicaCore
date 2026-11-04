"""No data fetch: test that one cherry-picked window cannot pass robust gate."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from v153.codec import Physical,Patch
from v161.protocol import MATRIX_KEYS
from v162.protocol import (
    check,WINDOW,WINDOWS,SCORED_TOKENS_EACH,NATIVE_ARMS,DOMAINS,
    WIKI_START,NEWS_START,NEWS_END,ORIGINAL_CODEBOOK_SHA,PARENT_MODEL_BYTES)
from v162.evidence import frozen_v161
from v162.audit import paired,summarize


def row(nll,kl,match=0):
    return {"tokens_scored":63,
        "sum_NLL":float(nll*63),
        "sum_teacher_NLL":float(4.2*63),
        "sum_teacher_to_student_KL":float(kl*63),
        "exact_FP32_next_token_argmax_count":match}


class OriginalPaidBlock0Robustness(unittest.TestCase):
    def test_predeclared_32plus32_disjoint_source_and_physical_parent(self):
        self.assertTrue(check())
        self.assertEqual((WINDOW,WINDOWS,SCORED_TOKENS_EACH),(64,32,2016))
        self.assertEqual((WIKI_START,NEWS_START,NEWS_END),(81920,25000,25500))
        self.assertEqual(PARENT_MODEL_BYTES,11292174)
        self.assertEqual(NATIVE_ARMS,("CB59_native","CB161_paid_original_block0"))

    def test_no_tail_outlier_can_fake_a_two_domain_pass(self):
        baseline=[row(15.,10.) for _ in range(WINDOWS)]
        lucky=[row(15.03,10.03) for _ in range(WINDOWS-1)]
        lucky.append(row(13.,8.))
        result=paired(lucky,baseline,"true_NLL")
        self.assertLess(result["mean_candidate_minus_CB59"],0)
        self.assertEqual(result["improved_windows_out_of_32"],1)
        self.assertGreater(result["after_excluding_single_strongest_improvement_mean_delta"],0)
        self.assertFalse(result["at_least_20_of_32_help_and_trimmed_mean_help"])
        measurements={
            arm:{d:list(baseline) for d in DOMAINS}
            for arm in NATIVE_ARMS}
        for d in DOMAINS:
            measurements[NATIVE_ARMS[1]][d]=list(lucky)
        times={arm:{d:100.0 for d in DOMAINS} for arm in NATIVE_ARMS}
        fail=summarize(measurements,times)
        self.assertFalse(fail["both_domain_precommitted_STRONG_NLL_KL_robustness_pass"])
        self.assertEqual(fail["per_domain_robustness"][DOMAINS[0]]
                         ["paired_TRUE_NLL"]["improved_windows_out_of_32"],1)
        better=[row(14.9,9.9) for _ in range(26)]+[row(15.1,10.1) for _ in range(6)]
        for d in DOMAINS:
            measurements[NATIVE_ARMS[1]][d]=list(better)
        passed=summarize(measurements,times)
        self.assertTrue(passed["both_domain_precommitted_STRONG_NLL_KL_robustness_pass"])
        self.assertEqual(passed["per_domain_robustness"][DOMAINS[1]]
                         ["teacher_top1_matches_new_out_of_2016"],0)
        measurements[NATIVE_ARMS[1]][DOMAINS[0]].pop()
        with self.assertRaises(ValueError):
            summarize(measurements,times)

    def test_verify_actual_seal_vs_complete_parent_sha_and_narrow_validation(self):
        full_sha="1"*64
        physical=Physical(b"original full CB59 only synthetic",0,{
            key:Patch(key,(64,64),np.zeros((32,2,8),dtype=np.float32),
                      np.zeros((64,),dtype=np.uint8))
            for key in MATRIX_KEYS
        },PARENT_MODEL_BYTES,full_sha)
        old={
            "stage":"v161_GENUINE_PAID_ORIGINAL_BLOCK0_REPAIR_VS_UNPAID_FP32_ORACLE",
            "unchanged_original_CB59_SHA256":ORIGINAL_CODEBOOK_SHA,
            "complete_source_SHA256":full_sha,
            "physically_paid_CB161_0p7bpp_source":{
                "complete_CB153_archive_bytes":PARENT_MODEL_BYTES,
                "all_in_bpp_unique_original":8*PARENT_MODEL_BYTES/134515008,
            },
            "per_arm_full_original_model_true_NLL_PPL_teacher_KL_top1":{
                "physically_paid_block0_improves_true_NLL_and_teacher_KL_in_both_domains":True,
                "new_VALID_frozen_effects":{
                    "wiki_block0_new_valid":{"paid_block0_minus_CB59_gold_NLL":-.39},
                    "news_block0_new_valid":{"paid_block0_minus_CB59_gold_NLL":-.005},
                }
            }
        }
        seal={
            "stage":"v161_PHYSICAL_SOURCE_LOCK",
            "pinned_original_CB59_SHA256":ORIGINAL_CODEBOOK_SHA,
            "physically_serialized_complete_CB161_SHA256":full_sha
        }
        with tempfile.TemporaryDirectory() as t:
            o=Path(t)/"real_v161.json"
            p=Path(t)/"frozen_before_valid.json"
            o.write_text(json.dumps(old))
            p.write_text(json.dumps(seal))
            self.assertEqual(frozen_v161(physical,o,p)["complete_v161_model_SHA256"],full_sha)
            seal["physically_serialized_complete_CB161_SHA256"]="0"*64
            p.write_text(json.dumps(seal))
            with self.assertRaises(ValueError):
                frozen_v161(physical,o,p)


if __name__=="__main__":
    unittest.main()
