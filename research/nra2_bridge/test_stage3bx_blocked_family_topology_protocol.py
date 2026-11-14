"""S3BX read-only anti-leakage checks; these are not independent EVAL tests."""
from __future__ import annotations
import copy
import unittest
from research.nra2_bridge.stage3bx_blocked_family_topology_protocol import (
    split_data, validate_splits, protocol, SPLITS, EXPECTED, CONSUMED_EVAL_RUNS,
    PRE_S3BR_PARENT_SHA, REJECTED_S3BT_SHA,
)


class S3BXBlockedCurriculumProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = split_data()
        cls.p = protocol(cls.groups)

    def test_exact_192_192_64_64_and_disjoint_hashes(self):
        self.assertTrue(validate_splits(self.groups))
        self.assertEqual({k: len(v) for k,v in self.groups.items()}, EXPECTED)
        hashes = [{r.prompt_sha256 for r in self.groups[k]} for k in SPLITS]
        for i in range(len(hashes)):
            for j in range(i+1, len(hashes)):
                self.assertTrue(hashes[i].isdisjoint(hashes[j]))
        self.assertEqual(sum(len(v) for v in hashes),512)

    def test_blocked_both_axes_not_accidentally_shared(self):
        g = [{r.topology_key for r in self.groups[k]} for k in SPLITS]
        self.assertEqual(len(g[0]),48)
        self.assertEqual(len(g[2]),16)
        self.assertEqual(g[0],g[1])
        self.assertEqual(g[2],g[3])
        self.assertTrue(g[0].isdisjoint(g[2]))
        for key in SPLITS:
            rows = self.groups[key]
            self.assertEqual(sum(bool(r.reachable) for r in rows), len(rows)//2)
            self.assertEqual(sum(r.query_style%2 == 0 for r in rows), len(rows)//2)
            self.assertEqual({r.query_style//2 for r in rows},
                             {0,1} if key in ("fit","topology_guard") else {2,3})

    def test_training_boundary_rejects_accidental_guard_leak(self):
        tampered = {key: list(rows) for key,rows in self.groups.items()}
        tampered["fit"][0] = tampered["double_guard"][0]
        with self.assertRaises(AssertionError):
            validate_splits(tampered)

    def test_pinned_parent_and_spent_eval_denied(self):
        self.assertEqual(self.p["safe_initialization_only_PRE_S3BR_S3BN_FP16_SHA256"],
                         PRE_S3BR_PARENT_SHA)
        self.assertEqual(self.p["rejected_do_not_initialize_S3BT_FP16_SHA256"],
                         REJECTED_S3BT_SHA)
        self.assertEqual(tuple(self.p["all_four_independent_EVAL128_already_CONSUMED_run_ids"]),
                         CONSUMED_EVAL_RUNS)
        self.assertTrue(self.p["future_training_must_only_load_fit_partition"])
        self.assertTrue(self.p["guard_partitions_not_allowed_in_gradient_or_alpha_or_checkpoint_choice"])
        self.assertTrue(self.p["these_512_were_used_by_rejected_S3BT_not_independent_EVAL"])
        self.assertTrue(self.p["no_old_EVAL_content_loaded_retested_or_retuned"])
        self.assertTrue(self.p["no_model_weights_CPU_features_training_or_scoring_this_step"])
        self.assertFalse(self.p["candidate_accepted"])
        self.assertEqual(self.p["output_gate"],0.0)

    def test_deterministic_sha_commitment(self):
        self.assertEqual(protocol(split_data())["commitment_sha256"],
                         self.p["commitment_sha256"])


if __name__ == "__main__":
    unittest.main()
