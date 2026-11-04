"""No new validation: deterministic 16-arm original layer0 diagnostic guards."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from v153.codec import Physical,Patch
from v161.protocol import MATRIX_KEYS
from v164.protocol import (check,FIT_DOMAINS,ARMS,SHORT,TOKENS_EACH,
    WINDOWS,WINDOW,EXPECTED_ARMS,PARENT_MODEL_BYTES)
from v164.arms import all_predeclared_arms
from v164.audit import summarize
from v164.evidence import verify_v162_fail
from v117.protocol import ORIGINAL_CB59_SHA


def fake_source():
    patches={key:Patch(key,(64,64),
        np.zeros((32,2,8),dtype=np.float32),
        np.zeros(64,dtype=np.uint8)) for key in MATRIX_KEYS}
    return Physical(b"ORIGINAL_FROZEN_CB59_SYNTHETIC",0,patches,
                    PARENT_MODEL_BYTES,"1"*64)


def row(nll,kl):
    return {"tokens_scored":63,
      "sum_NLL":float(63*nll),
      "sum_teacher_NLL":float(63*4.0),
      "sum_teacher_to_student_KL":float(63*kl),
      "exact_FP32_next_token_argmax_count":0}


class OriginalSevenOperatorAttribution(unittest.TestCase):
    def test_original_source_is_only_prior_train_fit_and_exact_16arms(self):
        self.assertTrue(check())
        self.assertEqual(EXPECTED_ARMS,16)
        self.assertEqual(len(ARMS),16)
        self.assertEqual(TOKENS_EACH,256)
        self.assertEqual((WINDOWS,WINDOW),(4,64))
        self.assertEqual(FIT_DOMAINS,("wiki_fit","news_fit"))
        self.assertEqual(len(MATRIX_KEYS),7)
        self.assertEqual(tuple(SHORT),("q_proj","k_proj","v_proj",
                                      "o_proj","gate_proj","up_proj","down_proj"))

    def test_ram_subsets_never_remove_physically_charged_parent_bits(self):
        paid=fake_source()
        arms=all_predeclared_arms(paid)
        self.assertEqual(tuple(arms),ARMS)
        self.assertEqual(len(arms["original_CB59"].patches),0)
        self.assertEqual(len(arms["paid_original_all7"].patches),7)
        for short in SHORT:
            self.assertEqual(len(arms[f"paid_ONLY_{short}"].patches),1)
            self.assertEqual(len(arms[f"paid_EXCEPT_{short}"].patches),6)
        self.assertEqual(len(paid.patches),7)
        self.assertEqual(paid.size,PARENT_MODEL_BYTES)
        with self.assertRaises(ValueError):
            all_predeclared_arms(Physical(paid.original,1,paid.patches,paid.size,paid.sha256))

    def test_reused_source_diagnostic_must_not_claim_two_domain_validation(self):
        rows={arm:{d:[row(15.0,11.0) for _ in range(WINDOWS)]
                   for d in FIT_DOMAINS} for arm in ARMS}
        for d in FIT_DOMAINS:
            rows["paid_ONLY_q_proj"][d]=[row(14.2,10.2) for _ in range(WINDOWS)]
            rows["paid_original_all7"][d]=[row(15.1,11.1) for _ in range(WINDOWS)]
        times={arm:{d:1.0 for d in FIT_DOMAINS} for arm in ARMS}
        answer=summarize(rows,times)
        self.assertEqual(answer["per_previous_original_TRAIN_FIT_source"][
            "wiki_fit"]["best_single_by_OLD_TRAIN_FIT_NLL_only_NOT_VALID"],
            MATRIX_KEYS[0])
        self.assertTrue(answer["this_is_mechanistic_hypothesis_NOT_quality_validation"])
        self.assertGreater(answer["per_previous_original_TRAIN_FIT_source"][
            "news_fit"]["all7_vs_original_CB59"][
            "delta_TRUE_next_token_NLL_vs_CB59"],0)
        rows["paid_original_all7"]["news_fit"].pop()
        with self.assertRaises(ValueError):
            summarize(rows,times)

    def test_v162_actual_news_failure_is_required_and_unaltered(self):
        source=fake_source()
        old={
          "stage":"v162_REAL_FULLY_PAID_BLOCK0_32WINDOW_TWO_DOMAIN_ROBUSTNESS",
          "source_parent_frozen_v161_SHA256":source.sha256,
          "original_complete_CB59_source_SHA256":ORIGINAL_CB59_SHA,
          "true_NLL_PPL_teacher_KL_early_teacher_argmax_results":{
            "both_domain_precommitted_STRONG_NLL_KL_robustness_pass":False,
            "per_domain_robustness":{
              "wikitext_fifth_new_block0":{"per_domain_robust_gate":True},
              "ag_news_fifth_new_block0":{
                "per_domain_robust_gate":False,
                "paired_TRUE_NLL":{"improved_windows_out_of_32":18},
                "paired_teacher_KL":{"improved_windows_out_of_32":17},
              },
            },
          },
        }
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/"v162.json"
            report.write_text(json.dumps(old))
            evidence=verify_v162_fail(source,report)
            self.assertEqual(evidence["v162_AG_News_gate"],"FAIL")
            old["true_NLL_PPL_teacher_KL_early_teacher_argmax_results"][
                "per_domain_robustness"]["ag_news_fifth_new_block0"][
                "paired_TRUE_NLL"]["improved_windows_out_of_32"]=22
            report.write_text(json.dumps(old))
            with self.assertRaises(ValueError):
                verify_v162_fail(source,report)

if __name__=="__main__":
    unittest.main()
