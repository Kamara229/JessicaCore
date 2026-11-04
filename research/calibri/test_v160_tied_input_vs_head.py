"""Frozen original input/head gain source and source VALID guard regression."""
import json
import tempfile
import unittest
from pathlib import Path

from v160.protocol import (check,GAINS,ARMS,DOMAINS,WINDOWS,WINDOW,
                           WIKI_START,NEWS_START,NEWS_END)
from v160.evidence import verify_v159
from v160.audit import summarize
from v159.protocol import CB153_SHA


def row(nll,kl):
    return {
       "tokens_scored":WINDOW-1,
       "sum_NLL":float(nll*(WINDOW-1)),
       "sum_teacher_NLL":float(4.0*(WINDOW-1)),
       "sum_teacher_to_student_KL":float(kl*(WINDOW-1)),
       "exact_FP32_next_token_argmax_count":0,
    }


class OriginalFullyPaidTiedInputStudy(unittest.TestCase):
    def test_exact_four_causal_arms_and_no_reused_train_validation(self):
        self.assertTrue(check())
        self.assertEqual(GAINS,((0.,0.),(1.,0.),(0.,1.),(1.,1.)))
        self.assertEqual(len(ARMS),4)
        self.assertEqual(len(DOMAINS),2)
        self.assertEqual((WIKI_START,NEWS_START,NEWS_END),
                         (65536,19000,19200))
        self.assertEqual(WINDOWS*WINDOW,512)

    def test_old_actual_embedding_drift_must_preexist_new_valid(self):
        old={
          "stage":"v159_ORIGINAL_30BLOCK_REAL_ACTIVATION_DRIFT_MAP",
          "full_physical_original_source_CB153_SHA256":CB153_SHA,
          "teacher_relative_RMSE_COSINE_and_stage_delta":{
            "existing_original_TRAIN_FIT_two_domain_activation_trace":{
              d:{
                "input_tied_embedding_original_CB59_RMSE":.72,
                "full_CB59_teacher_vs_packed_drift_by_source_stage":{
                  "block_00_output":{
                    "delta_relative_RMSE_vs_preceding_trace":.22
                  }
                },
              } for d in ("wiki_fit","news_fit")
            }
          },
        }
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"prior_v159.json"
            p.write_text(json.dumps(old))
            report=verify_v159(p)
            self.assertTrue(report["new_v160_validation_unseen_when_gains_locked"])
            old["teacher_relative_RMSE_COSINE_and_stage_delta"]["existing_original_TRAIN_FIT_two_domain_activation_trace"]["news_fit"]["input_tied_embedding_original_CB59_RMSE"]=.1
            p.write_text(json.dumps(old))
            with self.assertRaises(ValueError):
                verify_v159(p)

    def test_input_only_holdout_pass_requires_both_NLL_and_KL_and_two_domains(self):
        obs={arm:{domain:[row(15.,10.) for _ in range(WINDOWS)]
                  for domain in DOMAINS}
             for arm in ARMS}
        for d in DOMAINS:
            obs["paid_input_only_10"][d]=[row(14.8,9.8)
                                           for _ in range(WINDOWS)]
        result=summarize(obs)
        self.assertTrue(result[
            "paid_input_ONLY_improves_NLL_and_KL_on_both_new_domains"])
        obs["paid_input_only_10"][DOMAINS[1]]=[
            row(15.2,9.7) for _ in range(WINDOWS)]
        result=summarize(obs)
        self.assertFalse(result[
            "paid_input_ONLY_improves_NLL_and_KL_on_both_new_domains"])
        self.assertAlmostEqual(result[
            "paired_per_domain_causal_input_head_effects"][DOMAINS[1]]
            ["paid_input_only_10"]["vs_CB59_TRUE_next_token_NLL"],.2)
        obs["native_CB59_00"][DOMAINS[0]].pop()
        with self.assertRaises(ValueError):
            summarize(obs)

    def test_original_full_packed_source_format_claim_not_altered(self):
        from v160.protocol import ORIGINAL_V140_SHA,MAX_SOURCE_BYTES
        self.assertEqual(len(ORIGINAL_V140_SHA),64)
        self.assertLess(11672747+4,MAX_SOURCE_BYTES)


if __name__=="__main__":
    unittest.main()
