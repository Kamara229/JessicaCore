"""S3M graph-only preregistration tests; no 43M CPU or decoder loaded."""
import unittest

from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    FAMILIES, QUERY_STYLES, build_rows, graph_has_cycle, validate,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import (
    build_rows as previous_s3g, oracle,
)


class Stage3MGraphPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()

    def test_total_split_and_original_s3g_topology_disjoint(self):
        rows=self.rows
        self.assertEqual(len(rows),192)
        self.assertTrue(validate(rows))
        previous={tuple(sorted(row.edges)) for row in previous_s3g()}
        new={row.topology_key for row in rows}
        self.assertEqual(len(new),48)
        self.assertFalse(previous & new)

    def test_topology_disjoint_and_role_parity(self):
        rows=self.rows
        train={x.topology_key for x in rows if x.split=="TRAIN128"}
        dev={x.topology_key for x in rows if x.split=="DEV64"}
        self.assertEqual(len(train),32)
        self.assertEqual(len(dev),16)
        self.assertFalse(train&dev)
        for split,size in (("TRAIN128",128),("DEV64",64)):
            group=[x for x in rows if x.split==split]
            self.assertEqual(len(group),size)
            self.assertEqual(sum(x.reachable for x in group),size//2)
            for style in range(len(QUERY_STYLES)):
                subset=[x for x in group if x.query_style==style]
                self.assertEqual(sum(x.reachable for x in subset),len(subset)//2)

    def test_graph_family_and_independent_bfs(self):
        for row in self.rows:
            self.assertIn(row.family,FAMILIES)
            self.assertEqual(graph_has_cycle(row.edges),row.family=="cycle_sink")
            self.assertEqual(oracle(row.edges,row.src,row.dst),row.reachable)
            self.assertIn("R"+str(row.src),row.question)
            self.assertIn("R"+str(row.dst),row.question)

    def test_same_graph_has_opposite_answers_each_query_template(self):
        by_graph={}
        for row in self.rows:
            by_graph.setdefault(row.graph_id,[]).append(row)
        for rows in by_graph.values():
            self.assertEqual(len(rows),4)
            self.assertEqual(sorted(row.reachable for row in rows),[False,False,True,True])
            query_pairs={(r.src,r.dst) for r in rows}
            self.assertEqual(len(query_pairs),2)
            self.assertEqual({r.query_style for r in rows},{0,1,2,3})
            self.assertEqual(len({r.topology_key for r in rows}),1)

    def test_new_prompts_not_same_as_previous_control_prompts(self):
        old={r.prompt_sha256 for r in previous_s3g()}
        self.assertFalse(old & {r.prompt_sha256 for r in self.rows})


if __name__=="__main__":
    unittest.main()
