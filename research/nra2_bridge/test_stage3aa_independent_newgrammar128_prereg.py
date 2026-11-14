import unittest
from collections import Counter
from research.nra2_bridge.stage3aa_independent_newgrammar128_prereg import (
    build_rows,validate,commitment,GROUP,REVERSED_STYLES,
    S3Z_FP16_SHA256,S3T_PRIOR_SCORED_COMMIT_SHA,previous_rows,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets

class S3AAResearchPreregisterTests(unittest.TestCase):
    def test_128_new_text_and_32_disjoint_topologies(self):
        rows=build_rows()
        self.assertTrue(validate(rows))
        self.assertEqual(len(rows),128)
        self.assertEqual(len({r.topology_key for r in rows}),32)
        self.assertEqual(sum(r.reachable for r in rows),64)
        self.assertEqual(Counter(r.family for r in rows),
                         {"chain":32,"fork":32,"cycle_sink":32,"diamond":32})
        old=set(tuple(sorted(r.edges)) for block in previous_rows() for r in block)
        self.assertFalse(old & set(r.topology_key for r in rows))
    def test_two_reversed_two_forward_each_style_balanced(self):
        rows=build_rows()
        for row in rows:
            order=query_char_offsets(row.question)["text_order_nodes"]
            self.assertEqual(order[0]==row.dst,
                             row.query_style in REVERSED_STYLES)
        for style in range(4):
            part=[r for r in rows if r.query_style==style]
            self.assertEqual(len(part),32)
            self.assertEqual(sum(r.reachable for r in part),16)
    def test_commitments_deterministic_and_explicitly_no_scores(self):
        a=commitment(build_rows())
        b=commitment(build_rows())
        self.assertEqual(a,b)
        self.assertEqual(a["S3Z_role_head_physical_fp16_sha256_pinned"],
                         S3Z_FP16_SHA256)
        self.assertEqual(a["S3T_prior_scored_SHA"],
                         S3T_PRIOR_SCORED_COMMIT_SHA)
        self.assertTrue(a["new_real_43M_CPU_capture_not_started"])
        self.assertTrue(a["new_candidate_S3AA_predictions_not_inspected"])
        self.assertFalse(a["candidate_accepted"])
        self.assertEqual(a["gate"],0.0)
        self.assertTrue(a["public_research_precommit_not_cryptographically_hidden_blind"])
        self.assertTrue(a["no_additional_training_on_S3AA"])
if __name__=="__main__":
    unittest.main()
