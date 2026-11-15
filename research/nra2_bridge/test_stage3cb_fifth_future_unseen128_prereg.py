"""S3CB: immutable fifth independent128 prereg; NO inference or old EVAL reruns."""
from __future__ import annotations
import unittest
from research.nra2_bridge.stage3cb_fifth_future_unseen128_prereg import (
    build_rows,validate,commitment,STYLES,DESTINATION_FIRST,SOURCE_FIRST,
    ancestral_exclusions,S3BZ_SAFE_FP16_SHA,S3BU_FOURTH_COMMITMENT_SHA,
)


class S3CBFifthIndependent128Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()
        cls.commit=commitment(cls.rows)

    def test_16_truly_new_graphs_8_grammars(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual(len({r.topology_key for r in self.rows}),16)
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}),128)
        self.assertTrue({r.topology_key for r in self.rows}.isdisjoint(
            {r.topology_key for r in ancestral_exclusions()}))

    def test_balanced_grammars_truth_order(self):
        self.assertEqual(len(STYLES),len(set(STYLES)))
        self.assertEqual(sum(r.reachable for r in self.rows),64)
        self.assertEqual(sum(r.query_style in DESTINATION_FIRST for r in self.rows),64)
        for style in range(8):
            chunk=[r for r in self.rows if r.query_style==style]
            self.assertEqual(len(chunk),16)
            self.assertEqual(sum(r.reachable for r in chunk),8)

    def test_four_independent_sets_consumed_sha_frozen(self):
        c=self.commit
        self.assertEqual(c["S3BZ_safe_frozen_before_fifth_EVAL_sha256"],S3BZ_SAFE_FP16_SHA)
        self.assertEqual(c["S3BU_prior_fourth_consumed_commitment_sha256"],
                         S3BU_FOURTH_COMMITMENT_SHA)
        self.assertEqual(c["prior_first_independent_EVAL128_consumed_in"],43640363396)
        self.assertEqual(c["prior_second_independent_EVAL128_consumed_in"],43644153544)
        self.assertEqual(c["prior_third_independent_EVAL128_consumed_in"],43648038071)
        self.assertEqual(c["prior_fourth_independent_EVAL128_consumed_in"],43651233135)
        self.assertTrue(c["no_model_weights_or_Calibri_CPU_or_scores_or_training_this_stage"])
        self.assertEqual(c["output_gate"],0.)
        self.assertFalse(c["candidate_accepted"])

    def test_deterministic_commitment(self):
        self.assertEqual(commitment(build_rows())["commitment_sha256"],
                         self.commit["commitment_sha256"])


if __name__=="__main__":
    unittest.main()
