"""S3CG: sixth PRECOMMITTED independent128 graph+grammar data ONLY, no model."""
from __future__ import annotations
import unittest

from research.nra2_bridge.stage3cg_sixth_future_unseen128_prereg import (
    build_rows, validate, commitment, ancestral_exclusions,
    STYLES, DESTINATION_FIRST, SPLIT, S3CB_FIFTH_SHA,
    S3CF_SAFE_FP16_SHA, S3CF_TRAIN_SUCCESS_RUN,
)


class S3CGSixthIndependent128Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_rows()
        cls.committed = commitment(cls.rows)

    def test_16_fully_new_graphs_disjoint_all_five_old_evals(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual(len({r.topology_key for r in self.rows}),16)
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}),128)
        past=ancestral_exclusions()
        self.assertFalse({r.topology_key for r in self.rows} &
                         {tuple(sorted(r.edges)) for r in past})
        self.assertFalse({r.prompt_sha256 for r in self.rows} &
                         {r.prompt_sha256 for r in past})

    def test_four_graph_families_eight_novel_grammars_balanced(self):
        self.assertEqual(len(STYLES),8)
        self.assertEqual(len(set(STYLES)),8)
        self.assertEqual(sum(bool(r.reachable) for r in self.rows),64)
        self.assertEqual(sum(r.query_style in DESTINATION_FIRST for r in self.rows),64)
        for style in range(8):
            chunk=[r for r in self.rows if r.query_style==style]
            self.assertEqual(len(chunk),16)
            self.assertEqual(sum(r.reachable for r in chunk),8)

    def test_five_prior_evals_consumed_and_s3cf_fp16_locked_first(self):
        x=self.committed
        self.assertEqual(x["S3CB_prior_fifth_consumed_commitment_sha256"],S3CB_FIFTH_SHA)
        self.assertEqual(x["S3CF_safe_frozen_before_sixth_EVAL_sha256"],S3CF_SAFE_FP16_SHA)
        self.assertEqual(x["S3CF_successful_TRAIN_only_run"],S3CF_TRAIN_SUCCESS_RUN)
        self.assertEqual(x["S3CD_prior_fifth_consumed_run"],43677179638)
        self.assertTrue(x["all_five_previous_independent_EVAL128_permanently_consumed"])
        self.assertTrue(x["S3CF_candidate_physically_SHA_locked_before_sixth_EVAL_prereg"])
        self.assertTrue(x["no_model_weights_or_Calibri_CPU_or_scores_or_training_this_stage"])
        self.assertEqual(x["output_gate"],0.)
        self.assertFalse(x["candidate_accepted"])

    def test_reproducible_commitment(self):
        self.assertEqual(commitment(build_rows())["commitment_sha256"],
                         self.committed["commitment_sha256"])


if __name__=="__main__":
    unittest.main()
