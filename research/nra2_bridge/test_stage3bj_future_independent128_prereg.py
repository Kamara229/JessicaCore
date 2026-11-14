"""S3BJ source-only preregistration checks; no candidate is ever scored."""
import unittest
from research.nra2_bridge.stage3bj_future_independent128_prereg import (
    build_rows,validate,commitment,SPLIT,DESTINATION_FIRST,
)

class FutureBlind(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()
        cls.lock=commitment(cls.rows)

    def test_graph_and_prompt_disjoint(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual(len({r.topology_key for r in self.rows}),32)
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}),128)
        self.assertEqual({r.split for r in self.rows},{SPLIT})

    def test_balanced_direction_and_truth(self):
        self.assertEqual(sum(r.query_style in DESTINATION_FIRST for r in self.rows),64)
        self.assertEqual(sum(r.reachable for r in self.rows),64)
        self.assertEqual(self.lock["graph_count"],32)
        self.assertEqual(self.lock["count"],128)

    def test_exact_prereg_determinism(self):
        other=commitment(build_rows())
        self.assertEqual(other["commitment_sha256"],self.lock["commitment_sha256"])
        self.assertTrue(other["no_model_weights_or_CPU_capture_or_adaptor_scoring"])
        self.assertTrue(other["old_EVAL128_must_never_be_used_for_training_or_new_blind"])

if __name__=="__main__":
    unittest.main()
