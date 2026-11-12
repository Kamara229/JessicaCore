"""S179 integration contract tests, mock only; do not launch neural/GitHub CI."""
import unittest
from unittest.mock import patch

from calibri2.s179 import trace_integration as t
from calibri2.s179.trace_contract import S179TraceCollector


class S179IntegrationTests(unittest.TestCase):
    def test_original456_route_coverage_without_gold_leakage(self):
        old = {f"o{i}": {"question": f"unique old {i}", "expected": [99, 0]}
               for i in range(456)}
        new = {f"n{i}": {"question": f"unique new {i}", "expected": [98, 0]}
               for i in range(128)}
        old_ids, new_ids = sorted(old), sorted(new)
        routes = {key: "parent" for key in [*old_ids, *new_ids]}
        tok = type("Tok", (), {"eos_token_id": 0})()
        parent_old = {key: [1, 0] for key in old}
        parent_new = {key: [2, 0] for key in new}
        expected = (parent_old, parent_new, {"parent_fallback": 584}, [])
        capture = S179TraceCollector()
        with (patch.object(t, "ORIGINAL_AUDIT", return_value=expected),
              patch.object(t, "COLLECTOR", capture),
              patch.object(t, "FIRST_COHORT_DONE", False),
              patch.object(t, "CAPTURING", False)):
            result = t._traced_audit(
                None, old, new, old_ids, new_ids, routes, tok, None,
                None, None, None, None, parent_old, parent_new, None)
            self.assertEqual(result, expected)
            self.assertTrue(t.FIRST_COHORT_DONE)
            self.assertFalse(t.CAPTURING)
        self.assertEqual(len(capture.traces), 456)
        for row in capture.traces.values():
            self.assertEqual(row["full_generated_ids"], [1, 0])
            self.assertIsNone(row["conditional_step2"])
            self.assertNotIn("expected", row)
            self.assertFalse(row["gold_seen_at_inference"])

    def test_original_decoder_replayed_only_during_audit(self):
        events = []

        def frozen(*args):
            events.append("frozen")
            return [2, 0], {"parallel": True}

        with (patch.object(t, "ORIGINAL_DECODER", frozen),
              patch.object(t, "CAPTURING", False)):
            result = t._traced_decoder(
                None, None, None, None, None, None, "prompt",
                None, None, None)
        self.assertEqual(result[0], [2, 0])
        self.assertEqual(events, ["frozen"])


if __name__ == "__main__":
    unittest.main()
