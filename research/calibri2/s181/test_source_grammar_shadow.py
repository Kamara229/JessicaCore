"""S181: purely lexical synthetic tests; do NOT import model or Jessica."""
import ast
from pathlib import Path
import unittest

from calibri2.s181.source_grammar_shadow import (
    S30_TEMPLATES, S54_TEMPLATES, shadow_candidate, shadow_audit,
)


def _literal_tuple(path: Path, name: str, function_name: str | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if function_name is not None:
        nodes = next(x for x in nodes
                     if isinstance(x, ast.FunctionDef) and x.name == function_name).body
    for node in nodes:
        if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError("S181 pinned source literal not found")


class S181ShadowParserContracts(unittest.TestCase):
    def test_12_templates_exactly_match_source_literals(self):
        root = Path(__file__).resolve().parents[3]
        self.assertEqual(S30_TEMPLATES, _literal_tuple(
            root / "research/calibri2/s30/matched_dense_fp32_train.py",
            "patterns", "make_training_pairs"))
        self.assertEqual(S54_TEMPLATES, _literal_tuple(
            root / "research/calibri2/s54/diverse_train_preserve_S52_full_sequences.py",
            "NEW_TRAIN_FRAMES"))
        self.assertEqual((len(S30_TEMPLATES), len(S54_TEMPLATES)), (4, 8))

    def test_all_12_templates_with_two_owners_and_swapped_assignments(self):
        for template in (*S30_TEMPLATES, *S54_TEMPLATES):
            for left, right in [("red lamp", "red desk"), ("blue coat", "blue tree")]:
                for x, y in ((left, right), (right, left)):
                    with self.subTest(template=template[:22], left=x):
                        prompt = template.format(i=9000, a="OwnerA", b="OwnerB",
                                                 left=x, right=y)
                        self.assertEqual(shadow_candidate(prompt), x)

    def test_fails_closed_on_ambiguous_or_unknown_content(self):
        for template in (*S30_TEMPLATES, *S54_TEMPLATES):
            args = dict(i=9000, a="OwnerA", b="OwnerB",
                        left="red lamp", right="red desk")
            with self.subTest(template=template[:26]):
                self.assertIsNone(shadow_candidate(template.format(
                    **(args | {"a": "OwnerB"}))))
                self.assertIsNone(shadow_candidate(template.format(
                    **(args | {"right": "red lamp"}))))
                self.assertIsNone(shadow_candidate(template.format(
                    **(args | {"right": "blue desk"}))))
                self.assertIsNone(shadow_candidate(template.format(**args) + " extra"))
        self.assertIsNone(shadow_candidate("other question"))
        self.assertIsNone(shadow_candidate(""))
        self.assertIsNone(shadow_candidate(None))
        self.assertIsNone(shadow_candidate("x" * 1025))

    def test_repeated_owner_mismatch_fails_closed(self):
        template = S30_TEMPLATES[0]
        q = template.format(i=1, a="OwnerA", b="OwnerB",
                            left="red lamp", right="red desk")
        q = q.replace("State OwnerA's", "State OwnerC's")
        self.assertIsNone(shadow_candidate(q))

    def test_audit_has_no_answer_or_selector_fields(self):
        good = S54_TEMPLATES[0].format(
            i=9000, a="OwnerA", b="OwnerB",
            left="green leaf", right="green cup")
        result = shadow_audit([good, "not TRAIN"])
        self.assertEqual(result["candidate_emitted"], 1)
        self.assertEqual(result["abstained"], 1)
        self.assertTrue(result["no_gold_at_inference"])
        self.assertTrue(result["no_model_or_selector_changes"])
        self.assertTrue(result["not_accepted"])


if __name__ == "__main__":
    unittest.main()
