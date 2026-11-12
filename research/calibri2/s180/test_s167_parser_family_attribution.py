"""S180 frozen parser-scope tests: synthetic only, never run model."""
import unittest

from calibri2.s180.s167_parser_family_attribution import (
    FROZEN_FAMILIES, analyze_parser_scope)
from calibri2.s180.test_second_token_margin_posthoc import realistic_fixture


def fixture():
    trace, labels = realistic_fixture()
    family_sequence = [
        family for family, size in FROZEN_FAMILIES.items()
        for _ in range(size)
    ]
    assert len(family_sequence) == 456
    for i, family in enumerate(family_sequence):
        labels["all_456_per_identity_results"][i]["id"] = f"{family}/{i}/left"
    # Exercise one parser abstain among the original S82 known-template family.
    labels["all_456_per_identity_results"][0]["id"], labels["all_456_per_identity_results"][200]["id"] = (
        labels["all_456_per_identity_results"][200]["id"], labels["all_456_per_identity_results"][0]["id"])
    for i in list(range(17)) + [26]:
        trace["cases"][i]["guard_events"] = ["S167_owner_parser_abstained"]
    return trace, labels


class S180ParserScopeContracts(unittest.TestCase):
    def test_exact_frozen_family_coverage_and_18_second_misses(self):
        traces, labels = fixture()
        r = analyze_parser_scope(traces, labels)
        self.assertEqual(r["incorrect_total"], 28)
        self.assertEqual(r["second_token_misses"], 18)
        self.assertEqual(r["second_token_misses_parser_abstained"], 18)
        self.assertEqual(r["second_token_misses_inside_known_S164_family"], 1)
        self.assertEqual(r["second_token_misses_outside_known_S164_family"], 17)
        self.assertEqual(r["source_family_totals"]["S82_new"]["total"], 256)
        self.assertTrue(r["gold_never_used_in_inference"])
        self.assertFalse(r["model_accepted"])
        self.assertNotIn("target_ids", str(r))
        self.assertNotIn("full_generated_ids", str(r))

    def test_missing_parser_abstention_fails_closed(self):
        t, r = fixture()
        t["cases"][0]["guard_events"] = []
        with self.assertRaisesRegex(AssertionError, "18 token2 parser abstains"):
            analyze_parser_scope(t, r)

    def test_changed_family_id_fails_closed(self):
        t, r = fixture()
        r["all_456_per_identity_results"][0]["id"] = "unknown/0/left"
        with self.assertRaisesRegex(AssertionError, "source family"):
            analyze_parser_scope(t, r)

    def test_no_answer_errors_source_drift_fails_closed(self):
        t, r = fixture()
        t["cases"][0]["full_generated_ids"] = [10, 20, 0]
        with self.assertRaisesRegex(AssertionError, "source drift"):
            analyze_parser_scope(t, r)


if __name__ == "__main__":
    unittest.main()
