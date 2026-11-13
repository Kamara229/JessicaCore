"""Stage4 preregistered public examples: independent oracle and leakage checks.

Only validates publicly visible question/answer fixtures. Does NOT load Calibri,
NRA, Green Tunnel, train adapters, use QA/TEST, or score any model.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, deque
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
DATA = HERE / "stage4_hard_public_examples.jsonl"
FAMILIES = {
    "branching", "cycle", "reversed_direction", "counterfactual_edge",
    "forbidden_node", "two_questions", "conditional_edge",
    "shortest_weighted_path",
}


def reaches(edges, source, target, forbidden=None):
    if source == target or source == forbidden or target == forbidden:
        return False
    seen = {source}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for a, b in edges:
            if a == current and b != forbidden and b not in seen:
                if b == target:
                    return True
                seen.add(b)
                queue.append(b)
    return False


def cheapest(edges, weights, source, target):
    queue = [(0, source)]
    best = {source: 0}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != best[current]:
            continue
        if current == target:
            return str(cost)
        for (a, b), w in zip(edges, weights):
            if a == current and cost + w < best.get(b, float("inf")):
                best[b] = cost + w
                heapq.heappush(queue, (cost + w, b))
    return "none"


def has_cycle(edges):
    for source in range(6):
        next_nodes = [b for a, b in edges if a == source]
        seen = set()
        while next_nodes:
            node = next_nodes.pop()
            if node == source:
                return True
            if node in seen:
                continue
            seen.add(node)
            next_nodes.extend(b for a, b in edges if a == node)
    return False


def oracle(row):
    family = row["family"]
    d = row["oracle_inputs"]
    edges = [tuple(e) for e in row["edges"]]
    if family == "counterfactual_edge":
        removed = tuple(d["remove"])
        assert removed in edges, "Cannot remove nonexistent edge"
        edges = [e for e in edges if e != removed]
        new = tuple(d["add"])
        assert new not in edges, "Duplicate edit"
        edges.append(new)
    if family == "conditional_edge" and d["enabled"]:
        added = tuple(d["conditional"])
        assert added not in edges, "Duplicate conditional edge"
        edges.append(added)
    if family == "two_questions":
        return " ".join("yes" if reaches(edges, q["source"], q["target"])
                        else "no" for q in d["queries"])
    if family == "shortest_weighted_path":
        return cheapest(edges, d["weights"], d["source"], d["target"])
    result = reaches(edges, d["source"], d["target"],
                     d.get("forbidden") if family == "forbidden_node" else None)
    return "yes" if result else "no"


class Stage4PublicExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in DATA.read_text().splitlines()
                    if line.strip()]

    def test_count_unique_and_reserved_splits_not_misrepresented(self):
        self.assertEqual(len(self.rows), 48)
        self.assertEqual(set(r["family"] for r in self.rows), FAMILIES)
        self.assertEqual(set(Counter(r["family"] for r in self.rows).values()), {6})
        self.assertEqual(len(set(r["id"] for r in self.rows)), 48)
        self.assertEqual(len(set(r["question"] for r in self.rows)), 48)
        for row in self.rows:
            self.assertEqual(row["role"], "PUBLIC_EXAMPLE_NOT_BLIND")
            self.assertEqual(row["output_contract"],
                             "Exact answer followed by EOS; no decoder run yet.")
            self.assertNotIn(row["expected"], ("pass", "fail"))
            self.assertTrue(row["question"])
            self.assertTrue(all(a != b and 0 <= a < 6 and 0 <= b < 6
                                for a, b in row["edges"]))
            self.assertEqual(len({tuple(e) for e in row["edges"]}),
                             len(row["edges"]))

    def test_frozen_independent_graph_oracle(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(row["expected"], oracle(row))

    def test_no_answer_only_or_template_shortcuts(self):
        for family in FAMILIES - {"two_questions", "shortest_weighted_path"}:
            answers = {r["expected"] for r in self.rows if r["family"] == family}
            self.assertEqual(answers, {"yes", "no"}, family)
        self.assertEqual(
            {r["expected"] for r in self.rows if r["family"] == "two_questions"},
            {"yes no", "no yes"},
        )
        for row in self.rows:
            if row["family"] == "cycle":
                self.assertTrue(has_cycle(row["edges"]), row["id"])
            if row["family"] == "branching":
                outgoing = Counter(a for a, _ in row["edges"])
                self.assertGreaterEqual(max(outgoing.values()), 2)


if __name__ == "__main__":
    unittest.main()
