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

    def test_conditional_replay_matches_frozen_cpu_step2_logits(self):
        """A mock 2-step autoregressive forward must reuse original KV."""
        from types import SimpleNamespace
        import torch

        calls = []

        def fake_model(*, input_ids, past_key_values, use_cache,
                       output_hidden_states):
            calls.append((input_ids.tolist(), past_key_values))
            if past_key_values is None:
                logits = torch.tensor([[[0.0, 0.0, 5.0, 0.0],
                                        [0.0, 0.0, 5.0, 0.0]]])
                hidden = torch.zeros(1, 2, 2)
                cache = "frozen_kv"
            else:
                self.assertEqual(past_key_values, "frozen_kv")
                self.assertEqual(input_ids.tolist(), [[2]])
                logits = torch.tensor([[[3.0, 4.0, 1.0, 0.0]]])
                hidden = torch.zeros(1, 1, 2)
                cache = "next_kv"
            return SimpleNamespace(
                logits=logits, hidden_states=[hidden], past_key_values=cache)

        tok = type("Tok", (), {"eos_token_id": 0})()
        lm_head = torch.zeros(4, 2)
        transform = torch.nn.Identity()
        with (patch.object(t.s160, "prompt_tokens", return_value=[7, 8]),
              patch("v176.operators.clear", return_value=None)):
            scores = t._conditional_step2_replay(
                fake_model, object(), transform, transform, lm_head,
                tok, "prompt", None, [2, 1, 0])
        self.assertEqual(scores, [3.0, 4.0, 1.0, 0.0])
        self.assertEqual(len(calls), 2)

    def test_capture_does_not_substitute_scores_for_frozen_decoder(self):
        """Source and second-token observations may not alter the answer."""
        traces = {}
        with (patch.object(t, "ORIGINAL_DECODER",
                           return_value=([2, 1, 0],
                                         {"parallel": False,
                                          "neural_override_applied": True})),
              patch.object(t, "_conditional_step2_replay",
                           return_value=[0.3, 0.5, 1.0]),
              patch.object(t, "REPLAY_BY_SHA", traces),
              patch.object(t, "CAPTURING", True)):
            answer, route = t._traced_decoder(
                None, None, None, None, None, None, "original prompt",
                None, None, None)
        self.assertEqual(answer, [2, 1, 0])
        self.assertTrue(route["neural_override_applied"])
        self.assertEqual(len(traces), 1)
        self.assertEqual(next(iter(traces.values()))["conditional"],
                         [0.3, 0.5, 1.0])

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
