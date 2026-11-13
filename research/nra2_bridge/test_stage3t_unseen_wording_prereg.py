"""S3T no-score, new-grammar prereg tests (does not load model)."""
import hashlib
import json
import unittest
from research.nra2_bridge.stage3t_unseen_wording_prereg import (
    build_rows, validate, commitment, query_char_offsets, QUERY_STYLES,
    S3O_FROZEN_FP16_SHA256, FROZEN_THRESHOLD,
)
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    QUERY_STYLES as ORIGINAL_STYLES, build_rows as s3m_rows,
)
from research.nra2_bridge.stage3q_independent_soft_hard_prereg import build_rows as s3q_rows
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

class S3TNewGrammarPreregTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()

    def test_determinism_new_grammar_and_topology_disjoint(self):
        self.assertEqual(self.rows,build_rows())
        self.assertEqual(len(self.rows),64)
        self.assertEqual(len(set(QUERY_STYLES)),4)
        self.assertFalse(set(QUERY_STYLES)&set(ORIGINAL_STYLES))
        old={r.topology_key for r in s3m_rows()}
        old.update(r.topology_key for r in s3q_rows())
        self.assertFalse({r.topology_key for r in self.rows}&old)
        self.assertTrue(validate(self.rows))

    def test_s3o_query_clause_offset_grammar_and_inverse_roles(self):
        for r in self.rows:
            align=query_char_offsets(r.question)
            self.assertEqual(len(align["mention_spans"]),2)
            a,b=align["clause"]
            self.assertEqual(r.question[b-1],"?")
            self.assertEqual(set(align["text_order_nodes"]),{r.src,r.dst})
            for start,end in align["mention_spans"]:
                self.assertTrue(a<=start<end<=b)
            if r.query_style in (0,3):
                self.assertEqual(align["text_order_nodes"],[r.dst,r.src])
            else:
                self.assertEqual(align["text_order_nodes"],[r.src,r.dst])
            self.assertEqual(oracle(r.edges,r.src,r.dst),r.reachable)

    def test_balance_and_frozen_no_model_score_manifest(self):
        self.assertEqual(sum(r.reachable for r in self.rows),32)
        manifest=commitment(self.rows)
        self.assertEqual(manifest["questions"],64)
        self.assertEqual(manifest["graph_topologies"],16)
        self.assertEqual(manifest["S3O_frozen_adapter_fp16_sha256"],S3O_FROZEN_FP16_SHA256)
        self.assertEqual(manifest["edge_hard_threshold"],FROZEN_THRESHOLD)
        self.assertTrue(manifest["new_cpu_features_not_captured"])
        self.assertTrue(manifest["candidate_predicted_graph_scores_not_inspected"])
        self.assertTrue(manifest["not_sealed_final_blind"])
        self.assertFalse(manifest["candidate_accepted"])
        saved_sha=manifest.pop("commitment_sha256")
        self.assertEqual(saved_sha,hashlib.sha256(json.dumps(
            manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest())

if __name__=="__main__":
    unittest.main()
