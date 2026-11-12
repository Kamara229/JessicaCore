"""S179 trace API contract tests: no model test is run by this file alone."""
import json
import tempfile
import unittest
from pathlib import Path

from calibri2.s179.trace_contract import S179TraceCollector, posthoc_score


class TraceContractTests(unittest.TestCase):
    def test_conditional_logits_are_marked_as_ar_not_span_head(self):
        trace = S179TraceCollector()
        sha = trace.record(prompt="prompt A", route="S93", generated=[2, 3, 0],
                           eos_id=0, conditional_step2_logits=[1.0, -4.0, 5.0, 3.0],
                           guard_events=["S161_same_attribute_conflict_parent_fallback"],
                           source_votes_5=4, source_votes_10=8)
        row = trace.traces[sha]
        self.assertEqual(row["conditional_step2"]["eos_rank"], 3)
        self.assertEqual(row["conditional_step2"]["top5_token_ids_and_logits"][0][0], 2)
        self.assertFalse(row["gold_seen_at_inference"])
        self.assertNotIn("expected", json.dumps(row))
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "trace.json"
            trace.export(destination, expected_count=1)
            report = json.loads(destination.read_text())
            self.assertEqual(report["number"], 1)
            self.assertFalse(report["labels_used"])

    def test_non_s93_parent_route_has_no_fake_logits(self):
        traces = S179TraceCollector()
        sha = traces.record(prompt="prompt B", route="parent_only",
                            generated=[7, 0], eos_id=0,
                            conditional_step2_logits=None,
                            guard_events=["non_S93_parent_only"])
        self.assertIsNone(traces.traces[sha]["conditional_step2"])

    def test_duplicate_and_missing_coverage_fail(self):
        traces = S179TraceCollector()
        kw = dict(prompt="prompt C", route="S93", generated=[4, 0],
                  eos_id=0, conditional_step2_logits=None, guard_events=[])
        traces.record(**kw)
        with self.assertRaises(ValueError):
            traces.record(**kw)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(AssertionError):
                traces.export(Path(folder) / "trace.json")

    def test_label_comparison_is_delayed_until_after_inference(self):
        traces = S179TraceCollector()
        one = traces.record(prompt="prompt D", route="S93",
                            generated=[7, 5, 0], eos_id=0,
                            conditional_step2_logits=None, guard_events=[])
        two = traces.record(prompt="prompt E", route="parent_only",
                            generated=[4, 0], eos_id=0,
                            conditional_step2_logits=None, guard_events=[])
        r = posthoc_score(traces.traces, {one: [7, 6, 0], two: [4, 0]})
        self.assertEqual(r["full_eos_correct"], 1)
        self.assertEqual(r["incorrect_count"], 1)
        self.assertTrue(r["incorrect"][0]["second_token_error"])
        with self.assertRaises(AssertionError):
            posthoc_score(traces.traces, {one: [7, 6, 0]})


if __name__ == "__main__":
    unittest.main()
