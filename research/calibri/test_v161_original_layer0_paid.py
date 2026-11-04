"""Preregister original first-block paid source bitplane and oracle separation."""
import json
import tempfile
import unittest
from pathlib import Path

from v161.protocol import (
    check,ARMS,ORACLE_NAME,MATRIX_KEYS,DOMAINS,
    WIKI_START,NEWS_START,NEWS_END,WINDOW,WINDOWS,
    CB153_SHA,ORIGINAL_V140_SHA)
from v161.evidence import verify_old
from v161.audit import summarize


def row(nll,kl):
    return {
        "tokens_scored":63,
        "sum_NLL":float(63*nll),
        "sum_teacher_NLL":float(63*4.2),
        "sum_teacher_to_student_KL":float(63*kl),
        "exact_FP32_next_token_argmax_count":0,
    }


class FullyPaidOriginalBlock0(unittest.TestCase):
    def test_exact_7_source_2d_operators_and_fresh_dual_domain(self):
        self.assertTrue(check())
        self.assertEqual(len(MATRIX_KEYS),7)
        self.assertTrue(all(x.startswith("model.layers.0.") for x in MATRIX_KEYS))
        self.assertEqual(ORACLE_NAME,"UNPAID_FP32_block0_oracle")
        self.assertEqual((WIKI_START,NEWS_START,NEWS_END),(73728,22000,22200))
        self.assertEqual(WINDOW*WINDOWS,512)
        self.assertEqual(len(ARMS),3)

    def test_real_prior_v160_negative_input_result_is_required(self):
        v159={
            "stage":"v159_ORIGINAL_30BLOCK_REAL_ACTIVATION_DRIFT_MAP",
            "full_physical_original_source_CB153_SHA256":CB153_SHA,
            "teacher_relative_RMSE_COSINE_and_stage_delta":{
              "existing_original_TRAIN_FIT_two_domain_activation_trace":{
                domain:{
                  "full_CB59_teacher_vs_packed_drift_by_source_stage":{
                    "block_00_output":{
                      "delta_relative_RMSE_vs_preceding_trace":.228,
                    },
                  },
                } for domain in ("wiki_fit","news_fit")
              }
            },
        }
        v160={
            "stage":"v160_ORIGINAL_PAID_TIED_EMBEDDING_VS_OUTPUT_CAUSAL_VALID",
            "original_frozen_complete_CB140_source_SHA256":ORIGINAL_V140_SHA,
            "causal_input_vs_head_quality":{
              "paid_input_ONLY_improves_NLL_and_KL_on_both_new_domains":False,
              "paired_per_domain_causal_input_head_effects":{
                domain:{
                  "paid_input_only_10":{
                    "vs_CB59_TRUE_next_token_NLL":1.45,
                    "vs_CB59_FP32_teacher_KL":1.42,
                  }
                } for domain in ("wiki_fresh_embedding_causal",
                                 "news_fresh_embedding_causal")
              },
            },
        }
        with tempfile.TemporaryDirectory() as dir:
            a=Path(dir)/"prior159.json"
            b=Path(dir)/"prior160.json"
            a.write_text(json.dumps(v159))
            b.write_text(json.dumps(v160))
            evidence=verify_old(a,b)
            self.assertTrue(evidence[
                "v160_physical_input_only_correction_failed_true_NLL_and_KL_both_valid_domains"])
            v160["causal_input_vs_head_quality"]["paired_per_domain_causal_input_head_effects"][
                "news_fresh_embedding_causal"]["paid_input_only_10"][
                "vs_CB59_FP32_teacher_KL"]=-.1
            b.write_text(json.dumps(v160))
            with self.assertRaises(ValueError):
                verify_old(a,b)

    def test_only_genuinely_packed_arm_eligible_for_passing_gate(self):
        samples={
            arm:{domain:[row(15.,11.) for _ in range(WINDOWS)]
                 for domain in DOMAINS}
            for arm in ARMS
        }
        for domain in DOMAINS:
            samples[ARMS[1]][domain]=[row(14.9,10.9) for _ in range(WINDOWS)]
            samples[ORACLE_NAME][domain]=[row(10.,6.) for _ in range(WINDOWS)]
        quality=summarize(samples)
        self.assertTrue(quality[
            "physically_paid_block0_improves_true_NLL_and_teacher_KL_in_both_domains"])
        self.assertEqual(quality[
            "only_original_CB59_and_paid_block0_are_physically_eligible"],list(ARMS[:2]))
        samples[ARMS[1]][DOMAINS[1]]=[row(15.1,10.9) for _ in range(WINDOWS)]
        quality=summarize(samples)
        self.assertFalse(quality[
            "physically_paid_block0_improves_true_NLL_and_teacher_KL_in_both_domains"])
        samples[ARMS[0]][DOMAINS[0]].pop()
        with self.assertRaises(ValueError):
            summarize(samples)


if __name__=="__main__":
    unittest.main()
