"""Frozen source and negative robustness fixtures for original v_proj."""
import json
import tempfile
import unittest
from pathlib import Path
from v157.protocol import (check,WINDOWS,TOKENS,SCORED_TOKENS_PER_DOMAIN,
                           WIKI_START,NEWS_START,NEWS_END,ARMS,DOMAINS)
from v157.audit import (paired_audit,exact_sign_tail,deltas,summarize)
from v157.evidence import verify_prior_report
from v154.protocol import CB153_SHA


def row(nll,kl):
    return {
       "tokens_scored":63,"sum_NLL":nll*63,
       "sum_teacher_to_student_KL":kl*63,
    }


def agg(nll,kl):
    return {
        "tokens_scored":2016,
        "mean_TRUE_next_token_NLL":nll,
        "teacher_to_candidate_KL":kl,
        "teacher_next_token_argmax_agreement_fraction":0.,
    }


class SourceOutlierRobustness(unittest.TestCase):
    def test_fourth_sample_is_fixed_and_disjoint(self):
        self.assertTrue(check())
        self.assertEqual((WINDOWS,TOKENS,SCORED_TOKENS_PER_DOMAIN),
                         (32,2048,2016))
        self.assertEqual((WIKI_START,NEWS_START,NEWS_END),
                         (57344,17000,17400))
        self.assertEqual(len(ARMS),4)
        self.assertEqual(len(DOMAINS),2)

    def test_one_outlier_can_improve_mean_and_fail_strong_gate(self):
        baseline=[row(15.,10.) for _ in range(32)]
        improved=[row(15.03,10.03) for _ in range(31)]
        improved.append(row(13.,8.))
        nll=paired_audit(improved,baseline,"nll")
        kl=paired_audit(improved,baseline,"kl")
        self.assertLess(nll["all_32_window_mean"],0)
        self.assertLess(kl["all_32_window_mean"],0)
        self.assertEqual(nll["windows_improved"],1)
        self.assertGreater(nll["worst_single_favorable_window_dropped_mean"],0)
        self.assertFalse(nll["robust_enough"])
        self.assertFalse(kl["robust_enough"])
        self.assertEqual(exact_sign_tail(32,32),1/2**32)
        with self.assertRaises(ValueError):
            paired_audit(improved[:-1],baseline,"nll")
        with self.assertRaises(ValueError):
            deltas(improved,baseline,"invalid")

    def test_genuine_many_window_improvement_passes_predeclared_gate(self):
        baseline=[row(15.,10.) for _ in range(32)]
        alt=[row(14.8,9.8) for _ in range(26)]+[
            row(15.1,10.1) for _ in range(6)]
        self.assertTrue(paired_audit(alt,baseline,"nll")["robust_enough"])
        self.assertTrue(paired_audit(alt,baseline,"kl")["robust_enough"])
        records={arm:{domain:list(baseline) for domain in DOMAINS}
                 for arm in ARMS}
        scores={arm:{domain:agg(15,10) for domain in DOMAINS}
                for arm in ARMS}
        for domain in DOMAINS:
            records["paid_layer6_v_only"][domain]=list(alt)
            scores["paid_layer6_v_only"][domain]=agg(14.85625,9.85625)
        actual=summarize(scores,records)
        self.assertTrue(actual["v_proj_STRONG_two_domain_robustness_pass"])
        records["paid_layer6_v_only"][DOMAINS[0]].pop()
        with self.assertRaises(ValueError):
            summarize(scores,records)

    def test_preregister_requires_the_actual_v156_wiki_3of8_and_2of8(self):
        previous={
            "stage":"v156_POST_SELECTION_THIRD_DISJOINT_SOURCE_VALID_VPROJ",
            "immutable_whole_CB153_sha":CB153_SHA,
            "paired_window_sign_and_holdout_audit":{
                "v_proj_improves_NLL_and_FP32_teacher_KL_on_both_third_holdout_domains":True,
                "new_domain_deltas_and_pairing":{},
            },
        }
        for domain in ("wiki_THIRD_unseen_VALID","news_THIRD_unseen_VALID"):
            prior={
                "v_proj_minus_CB59_NLL":-.1,
                "v_proj_minus_CB59_KL":-.1,
                "paired_eight_windows":{
                    "paired_window_mean_NLL_deltas":[-.1]*8,
                    "NLL_better_windows_of_8":3,
                    "KL_better_windows_of_8":2,
                }
            }
            previous["paired_window_sign_and_holdout_audit"]["new_domain_deltas_and_pairing"][domain]=prior
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"v156_prior.json"
            p.write_text(json.dumps(previous))
            self.assertTrue(verify_prior_report(p)["v156_is_model_selection_evidence_NOT_final"])
            previous["paired_window_sign_and_holdout_audit"]["new_domain_deltas_and_pairing"]["wiki_THIRD_unseen_VALID"]["paired_eight_windows"]["NLL_better_windows_of_8"]=8
            p.write_text(json.dumps(previous))
            with self.assertRaises(ValueError):
                verify_prior_report(p)


if __name__=="__main__":
    unittest.main()
