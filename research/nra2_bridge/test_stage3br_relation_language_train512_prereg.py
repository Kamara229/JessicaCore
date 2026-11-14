"""S3BR new directional relation TRAIN512 only; never score spent EVAL sets."""
from __future__ import annotations
import unittest
from research.nra2_bridge.stage3br_relation_language_train512_prereg import (
    build_rows,commitment,validate,STYLES,SOURCE_FIRST,DESTINATION_FIRST,
)

class S3BRNewTrainingOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()
        cls.report=commitment(cls.rows)

    def test_64_only_original_train_topologies(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual(len({r.topology_key for r in self.rows}),64)
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}),512)

    def test_new_directional_grammar_vs_label_balance(self):
        self.assertEqual(len(set(STYLES)),8)
        self.assertEqual(sum(r.reachable for r in self.rows),256)
        self.assertEqual(sum(r.query_style in DESTINATION_FIRST for r in self.rows),256)
        for style in range(8):
            g=[r for r in self.rows if r.query_style==style]
            self.assertEqual(len(g),64)
            self.assertEqual(sum(r.reachable for r in g),32)

    def test_spent_independent_holdout_exclusion(self):
        report=self.report
        self.assertEqual(report["prior_EVAL1_consumed_run"],43640363396)
        self.assertEqual(report["prior_EVAL2_consumed_run"],43644153544)
        self.assertEqual(report["prior_EVAL3_consumed_run"],43648038071)
        self.assertTrue(report["prior_EVALs_NEVER_model_input_nor_target_for_train"])
        self.assertTrue(report["source_original_TRAIN_graphs_only"])
        self.assertTrue(report["no_cpu_tokens_or_weights_or_training_or_scoring_this_step"])
        self.assertEqual(report["output_gate"],0.0)
        self.assertFalse(report["candidate_accepted"])

    def test_deterministic_commitment(self):
        self.assertEqual(commitment(build_rows())["commitment_sha256"],self.report["commitment_sha256"])

if __name__=="__main__":
    unittest.main()
