"""S38 new adversarial Russian source-only policy diagnostic, NO neural inference."""
from __future__ import annotations
import inspect
import unittest

from research.nra3_s20.s38_novel_russian_structure_challenge_source import (
    ROWS,STAGE,validate_challenge,challenge_digest,
)
from research.nra3_s20.s38_novel_russian_structure_rule_only_audit import (
    score_policy,raw_input_only_batch,all_candidate_upper_bound,
    CHALLENGE_FROZEN_SOURCE_BLOB_GIT_SHA,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import text_only_inputs

class S38StructureAdversarialPolicies(unittest.TestCase):
    def test_frozen_32_novel_examples_balanced_by_structure_not_used_in_TRAIN256(self):
        report=validate_challenge()
        self.assertEqual(report["count"],32)
        self.assertEqual(report["positive"],24)
        self.assertEqual(report["negative"],8)
        self.assertEqual(set(report["group_counts"].values()),{8})
        self.assertEqual(report["canonical_source_and_label_sha256"],challenge_digest())
        self.assertEqual(CHALLENGE_FROZEN_SOURCE_BLOB_GIT_SHA,
                         "d1c5be4402fb456adab896a828e742f5f34c48dd")
        self.assertEqual(set(r.text for r in ROWS)&set(text_only_inputs()),set())
        self.assertEqual(STAGE,
                         "NRA3_S38_NEW_POSTHOC_RUSSIAN_NEGATION_CHALLENGE_SOURCE32_V1")

    def test_input_only_no_43m_fake_hidden_or_gold_in_rule_forward(self):
        x,valid,words=raw_input_only_batch()
        self.assertEqual(x.shape[0],32)
        self.assertEqual(x.shape[-1],338)
        self.assertEqual(int(x[...,:320].count_nonzero()),0)
        self.assertEqual(int(valid.sum()),sum(len(t) for t in words))
        self.assertEqual(len(all_candidate_upper_bound(*valid.shape).argument_polarity.shape),4)
        # The source annotations are referenced only AFTER both decoders.
        src=inspect.getsource(score_policy)
        first=src.index("s35=apply_local_ne_dative_scope")
        second=src.index("scope37=apply_raw_source_verb_scope")
        offline=src.index("for i,r in enumerate(rows):")
        self.assertLess(first,second)
        self.assertLess(second,offline)
        self.assertIn("actual_model_not_present" if False else
                      "all_candidate_upper_bound",src)
        self.assertNotIn("real_j7_baseline(",src)
        self.assertNotIn("new_graph_model(",src)

    def test_adversarial_new_structure_rule_results_are_diagnostic_not_success_gate(self):
        result=score_policy()
        self.assertEqual(result["frozen_S38_source_sha256"],challenge_digest())
        self.assertEqual(result["S35_eligibility_counts"]["total"],32)
        self.assertEqual(result["S37_eligibility_counts"]["total"],32)
        self.assertTrue(result["all_candidates_have_deliberately_saturated_FAKE_REJECTED_logits"])
        self.assertTrue(result["scores_are_policy_eligibility_NOT_NEURAL_precision_recall"])
        self.assertTrue(result["future_real43M_test_requires_new_separately_frozen_CPU_capture"])
        self.assertTrue(result["genuine_original43M_GT_embeddings_NOT_computed_on_S38"])
        self.assertFalse(result["candidate_accepted"])
        self.assertFalse(result["activation_allowed"])
        for group in result["by_predeclared_new_structure"].values():
            self.assertEqual(group["count"],8)
        self.assertGreater(
            result["S37_eligibility_counts"]["true_rejected_pairs_blocked"],0,
            "adversarial fixed challenge must expose new morphology/negation false negatives")


if __name__=="__main__":
    unittest.main()
