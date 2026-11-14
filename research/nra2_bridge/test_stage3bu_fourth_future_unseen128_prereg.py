"""S3BU FOURTH independent128 data-only immutability; NO neural model or CPU calls."""
from __future__ import annotations
import unittest
from research.nra2_bridge.stage3bu_fourth_future_unseen128_prereg import (
    build_rows,commitment,validate,SPLIT,DESTINATION_FIRST,STYLES,
)

class S3BUFourthIndependentNoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()
        cls.lock=commitment(cls.rows)

    def test_no_old_graph_or_prompt_overlap(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual({r.split for r in self.rows},{SPLIT})
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}),128)
        self.assertEqual(len({r.topology_key for r in self.rows}),16)

    def test_new_grammar_and_balanced_direction(self):
        self.assertEqual(len(set(STYLES)),8)
        self.assertEqual(sum(r.reachable for r in self.rows),64)
        self.assertEqual(sum(r.query_style in DESTINATION_FIRST for r in self.rows),64)
        for j in range(8):
            group=[r for r in self.rows if r.query_style==j]
            self.assertEqual(len(group),16)
            self.assertEqual(sum(r.reachable for r in group),8)

    def test_protected_three_consumed_and_fourth_future(self):
        a=self.lock
        self.assertEqual(a["prior_first_independent_EVAL128_consumed_in"],43640363396)
        self.assertEqual(a["prior_second_independent_EVAL128_consumed_in"],43644153544)
        self.assertEqual(a["prior_third_independent_EVAL128_consumed_in"],43648038071)
        self.assertTrue(a["ALL_previous_independent_EVALs_excluded_from_optimizer_and_candidate_selection"])
        self.assertTrue(a["S3BT_candidate_physically_SHA_locked_before_first_fourth_EVAL_CPU"])
        self.assertTrue(a["no_model_weights_or_Calibri_CPU_or_scores_or_training_this_stage"])
        self.assertFalse(a["candidate_accepted"])
        self.assertEqual(a["output_gate"],0.0)

    def test_deterministic_protocol_without_scores(self):
        self.assertEqual(commitment(build_rows())["commitment_sha256"],self.lock["commitment_sha256"])

if __name__=="__main__":
    unittest.main()
