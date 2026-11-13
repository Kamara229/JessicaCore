"""S3C relation label correctness and genuine structural split invariants."""
import unittest
import torch

from research.nra2_bridge.stage3c_graph_data import (
    build_graph_rows,build_graph_targets,validate_graph_rows,reachable,
    no_answer_template_lookup,TRAIN_COUNT,HELD_COUNT,NODE_COUNT,
)


class Stage3CGraphDataTests(unittest.TestCase):
    def test_declared_48_unique_structurally_disjoint(self):
        rows=build_graph_rows()
        self.assertEqual((len(rows[:TRAIN_COUNT]),len(rows[TRAIN_COUNT:])),(32,16))
        self.assertEqual([r.prompt_sha for r in rows],
                         [r.prompt_sha for r in build_graph_rows()])
        self.assertTrue(no_answer_template_lookup(rows))
        self.assertEqual(sum(r.reachable for r in rows[:32]),16)
        self.assertEqual(sum(r.reachable for r in rows[32:]),8)
        self.assertTrue(all(r.active_count<6 for r in rows[:32]))
        self.assertTrue(all(r.active_count==6 for r in rows[32:]))
        self.assertFalse({r.edges for r in rows[:32]}&{r.edges for r in rows[32:]})

    def test_physical_graph_labels_and_reachability(self):
        rows=build_graph_rows()
        tensors=build_graph_targets(rows)
        self.assertEqual(tuple(tensors["adjacency"].shape),(48,6,6))
        self.assertEqual(tuple(tensors["source"].shape),(48,6))
        self.assertEqual(tuple(tensors["target"].shape),(48,6))
        self.assertEqual(tuple(tensors["reachability"].shape),(48,))
        self.assertTrue(torch.equal(tensors["source"].sum(1),torch.ones(48,dtype=torch.float16)))
        self.assertTrue(torch.equal(tensors["target"].sum(1),torch.ones(48,dtype=torch.float16)))
        for i,r in enumerate(rows):
            edge_map={(int(a),int(b)) for a,b in
                      torch.nonzero(tensors["adjacency"][i]).tolist()}
            self.assertEqual(edge_map,set(r.edges))
            self.assertEqual(int(tensors["reachability"][i]),int(r.reachable))

    def test_graph_direction_matters_not_prompt_order(self):
        rows=build_graph_rows()
        for r in rows:
            flipped=tuple((b,a) for a,b in r.edges)
            self.assertNotEqual(reachable(r.edges,r.source,r.target),
                                reachable(flipped,r.source,r.target))
        validate_graph_rows(rows)


if __name__=="__main__":
    unittest.main()
