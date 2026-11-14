"""S3BO third independent EVAL128 prereg tests (data only, zero model)."""
from __future__ import annotations
import unittest
from research.nra2_bridge.stage3bo_third_future_independent128_prereg import (
    build_rows,validate,commitment,SPLIT,DESTINATION_FIRST,S3AQ_SHA,S3BJ_SHA,
)

class ThirdBlindPreregTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()
        cls.lock=commitment(cls.rows)

    def test_new_topologies_and_prompt_count(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual({r.split for r in self.rows},{SPLIT})
        self.assertEqual(len({r.topology_key for r in self.rows}),16)
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}),128)

    def test_balanced_role_direction_styles_and_truth(self):
        self.assertEqual(sum(r.query_style in DESTINATION_FIRST for r in self.rows),64)
        self.assertEqual(sum(r.reachable for r in self.rows),64)
        for style in range(8):
            items=[r for r in self.rows if r.query_style==style]
            self.assertEqual(len(items),16)
            self.assertEqual(sum(r.reachable for r in items),8)

    def test_commitment_has_old_consumed_holdout_exclusion(self):
        p=self.lock
        self.assertEqual(p["prior_original_S3AQ_SHA256"],S3AQ_SHA)
        self.assertEqual(p["prior_S3BJ_SHA256"],S3BJ_SHA)
        self.assertTrue(p["no_spent_EVALs_reused_as_NEW_blind_or_fit"])
        self.assertTrue(p["no_model_weights_original_CPU_inference_or_adapter_scoring"])
        self.assertTrue(p["S3BN_frozen_candidate_SHA_required_before_CPU_capture_or_scores"])
        self.assertFalse(p["candidate_accepted"])
        self.assertEqual(p["output_gate"],0.0)

    def test_immutable_reproducible_precommit(self):
        q=commitment(build_rows())
        self.assertEqual(q["commitment_sha256"],self.lock["commitment_sha256"])

if __name__=="__main__":
    unittest.main()
