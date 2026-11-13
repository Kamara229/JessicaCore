"""S3Q: pure preregistration tests only; no model loaded or DEV scores read."""
import hashlib
import json
import unittest

from research.nra2_bridge.stage3q_independent_soft_hard_prereg import (
    GROUP,FAMILIES,QUERY_STYLES,HEADERS,TOTAL_GRAPHS,TOTAL_PROMPTS,
    SEED,S3O_FP16_SHA256,FROZEN_THRESHOLD,build_rows,validate,commitment,
)
from research.nra2_bridge.stage3m_role_balanced_graph_data import build_rows as s3m
from research.nra2_bridge.stage3g_factorized_graph_controls import build_rows as s3g,oracle


class S3QIndependentGraphTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()

    def test_unique_reproducible_predeclared_topologies(self):
        self.assertEqual(self.rows,build_rows())
        self.assertEqual(len(self.rows),64)
        self.assertTrue(validate(self.rows))
        newer={r.topology_key for r in self.rows}
        old={r.topology_key for r in s3m()}
        old.update(tuple(sorted(r.edges)) for r in s3g())
        self.assertEqual(len(newer),16)
        self.assertFalse(newer & old)
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}),64)
        self.assertFalse({r.prompt_sha256 for r in self.rows}&
                         {r.prompt_sha256 for r in s3m()})

    def test_no_family_question_header_length_label_shortcuts(self):
        self.assertEqual(sum(r.reachable for r in self.rows),32)
        for family in FAMILIES:
            group=[r for r in self.rows if r.family==family]
            self.assertEqual(len(group),16)
            self.assertEqual(sum(r.reachable for r in group),8)
            for style in range(len(QUERY_STYLES)):
                parts=[r for r in group if r.query_style==style]
                self.assertEqual(len(parts),4)
                self.assertEqual(sum(r.reachable for r in parts),2)
        for style in range(4):
            for n in (4,5,6):
                sub=[r for r in self.rows if r.query_style==style and r.active_nodes==n]
                self.assertTrue(sub)
                self.assertEqual(2*sum(r.reachable for r in sub),len(sub))
            for header in HEADERS:
                pre=header.partition("{edges}")[0]
                sub=[r for r in self.rows if r.query_style==style and r.question.startswith(pre)]
                self.assertTrue(sub)
                self.assertEqual(2*sum(r.reachable for r in sub),len(sub))

    def test_each_graph_has_2_yes_2_no_reverse_directions(self):
        by_id={}
        for r in self.rows:
            self.assertEqual(r.split,GROUP)
            self.assertEqual(oracle(r.edges,r.src,r.dst),r.reachable)
            by_id.setdefault(r.graph_id,[]).append(r)
        self.assertEqual(len(by_id),16)
        for g in by_id.values():
            self.assertEqual(len(g),4)
            self.assertEqual(sum(r.reachable for r in g),2)
            self.assertEqual({r.query_style for r in g},{0,1,2,3})
            pairs={(r.src,r.dst) for r in g}
            self.assertEqual(len(pairs),2)
            self.assertTrue(all((b,a) in pairs for a,b in pairs))

    def test_frozen_training_independence_and_SHA_commitment(self):
        c=commitment(self.rows)
        self.assertEqual(c["seed"],SEED)
        self.assertEqual(c["prompts"],64)
        self.assertEqual(c["topologies"],16)
        self.assertEqual(c["S3O_frozen_sha256"],S3O_FP16_SHA256)
        self.assertEqual(c["predeclared_edge_rule"],
                         "hard_edge = (sigmoid(edge_logit) > 0.5).float()")
        self.assertEqual(FROZEN_THRESHOLD,0.5)
        self.assertTrue(c["S3Q_predictions_not_inspected"])
        self.assertTrue(c["NRA_model_not_retrained"])
        self.assertFalse(c["candidate_accepted"])
        self.assertEqual(len(c["question_SHA256_in_order"]),64)
        self.assertEqual(len(c["topology_SHA256_in_order"]),16)
        original=c.pop("commitment_sha256")
        self.assertEqual(original,hashlib.sha256(json.dumps(
            c,sort_keys=True,separators=(",",":")).encode()).hexdigest())


if __name__=="__main__":
    unittest.main()
