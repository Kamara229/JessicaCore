"""Synthetic deterministic safeguards, no actual v156 QA/VALID source fetched."""
import json
import tempfile
import unittest
from pathlib import Path
from v156.protocol import (check,SELECTED_KEY,SECONDARY_KEY,ARMS,DOMAINS,
                           WIKI_START,NEWS_START,NEWS_END,TOKENS)
from v156.selection import freeze_exploratory_evidence
from v156.audit import paired_window_stats,exact_sign_tail,summarize
from v154.protocol import CB153_SHA


def per_window(nll,kl):
    return {"tokens_scored":63,
            "sum_NLL":float(nll*63),
            "sum_teacher_to_student_KL":float(kl*63)}


class FrozenThirdHoldoutTest(unittest.TestCase):
    def test_all_registered_new_sources_AND_candidate(self):
        self.assertTrue(check())
        self.assertEqual(SELECTED_KEY,"model.layers.6.self_attn.v_proj.weight")
        self.assertEqual(SECONDARY_KEY,"model.layers.6.self_attn.o_proj.weight")
        self.assertEqual((WIKI_START,NEWS_START,NEWS_END,TOKENS),
                         (49152,14300,14400,512))
        self.assertEqual(len(ARMS),4)
        self.assertEqual(len(DOMAINS),2)

    def test_exact_paired_window_sign_tail(self):
        self.assertEqual(exact_sign_tail(8,8),1/256)
        self.assertEqual(exact_sign_tail(4,8),163/256)
        baseline=[per_window(14,10) for _ in range(8)]
        candidate=[per_window(13,9) for _ in range(8)]
        scores=paired_window_stats(candidate,baseline)
        self.assertEqual(scores["NLL_better_windows_of_8"],8)
        self.assertEqual(scores["KL_better_windows_of_8"],8)
        self.assertAlmostEqual(scores["paired_window_mean_NLL_deltas"][0],-1)
        with self.assertRaises(ValueError):
            paired_window_stats(candidate[:-1],baseline)
        with self.assertRaises(ValueError):
            exact_sign_tail(9,8)

    def test_prior_v155_real_negative_control_promotion_validation(self):
        control={
            "stage":"v155_BEST_SINGLE_PAID_ORIGINAL_OPERATOR_UNSEEN_TWO_DOMAIN_VALID",
            "unchanged_frozen_original_complete_CB153":{
                "whole_source_SHA256":CB153_SHA,
            },
            "selection_FROM_prior_v154_eight_reused_prompts_ONLY":{
                "best":SECONDARY_KEY,"worst":SELECTED_KEY,
            },
            "one_operator_vs_original_CB59_and_all7":{
                "wiki_new_single_op_valid":{},
                "news_new_single_op_valid":{},
            },
            "source_validation_true_NLL_KL_PPL_and_top1":{
                "CB59_original":{},
                "v154_worst_single":{},
            },
        }
        for domain in ("wiki_new_single_op_valid","news_new_single_op_valid"):
            control["source_validation_true_NLL_KL_PPL_and_top1"]["CB59_original"][domain]={
                "mean_TRUE_next_token_NLL":15.,
                "teacher_to_candidate_KL":11.,
            }
            control["source_validation_true_NLL_KL_PPL_and_top1"]["v154_worst_single"][domain]={
                "mean_TRUE_next_token_NLL":14.9,
                "teacher_to_candidate_KL":10.9,
            }
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"v155_source_result.json"
            p.write_text(json.dumps(control))
            evidence=freeze_exploratory_evidence(p)
            self.assertTrue(evidence["new_third_VALID_not_used_for_v_proj_selection"])
            control["source_validation_true_NLL_KL_PPL_and_top1"]["v154_worst_single"]["wiki_new_single_op_valid"]["teacher_to_candidate_KL"]=11.1
            p.write_text(json.dumps(control))
            with self.assertRaises(ValueError):
                freeze_exploratory_evidence(p)

    def test_mixed_third_holdout_failure_visible(self):
        def agg(nll,kl):
            return {"mean_TRUE_next_token_NLL":nll,
                    "teacher_to_candidate_KL":kl,
                    "teacher_next_token_argmax_agreement_fraction":0}
        summary={arm:{domain:agg(15,11) for domain in DOMAINS}
                 for arm in ARMS}
        raw={arm:{domain:[per_window(15,11) for _ in range(8)]
                  for domain in DOMAINS} for arm in ARMS}
        summary["paid_v_proj_only"][DOMAINS[0]]=agg(14,10)
        summary["paid_v_proj_only"][DOMAINS[1]]=agg(16,10)
        raw["paid_v_proj_only"][DOMAINS[0]]=[per_window(14,10) for _ in range(8)]
        raw["paid_v_proj_only"][DOMAINS[1]]=[per_window(16,10) for _ in range(8)]
        report=summarize(summary,raw)
        self.assertFalse(report["v_proj_improves_NLL_and_FP32_teacher_KL_on_both_third_holdout_domains"])
        self.assertGreater(report["new_domain_deltas_and_pairing"][DOMAINS[1]]["v_proj_minus_CB59_NLL"],0)
        raw["native_CB59"][DOMAINS[0]].pop()
        with self.assertRaises(ValueError):
            summarize(summary,raw)


if __name__=="__main__":
    unittest.main()
