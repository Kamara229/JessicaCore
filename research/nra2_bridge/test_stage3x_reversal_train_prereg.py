import unittest
from research.nra2_bridge.stage3x_reversal_train_prereg import build_rows,commitment,validate,COUNT,GRAPH_COUNT,REVERSED_STYLES
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
class Stage3XPrecommitTests(unittest.TestCase):
    def test_128_train_graphs_and_yes_no(self):
        rows=build_rows()
        self.assertEqual(len(rows),COUNT)
        self.assertEqual(len({r.topology_key for r in rows}),GRAPH_COUNT)
        self.assertEqual(sum(r.reachable for r in rows),64)
        self.assertTrue(validate(rows))
    def test_only_new_wordings_two_reversed(self):
        rows=build_rows()
        for r in rows:
            o=query_char_offsets(r.question)
            self.assertEqual(o["text_order_nodes"][0]==r.dst,r.query_style in REVERSED_STYLES)
    def test_sha_determinism_pretrained_no_scoring(self):
        a=commitment(build_rows())
        b=commitment(build_rows())
        self.assertEqual(a,b)
        self.assertTrue(a["new_training_not_started"])
        self.assertEqual(a["output_gate"],0)
        self.assertFalse(a["candidate_accepted"])
if __name__=="__main__":
    unittest.main()
