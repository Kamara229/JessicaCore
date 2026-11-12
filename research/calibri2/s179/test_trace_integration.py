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

    def _frozen_mock_decoder(self, *, second_scores, token2, override=False):
        """Original two cached model forwards; any S179 extra forward is a bug."""
        from types import SimpleNamespace
        import torch

        calls = []
        class FrozenModel(torch.nn.Module):
            def forward(self, *, input_ids, past_key_values, use_cache,
                        output_hidden_states):
                calls.append((input_ids.tolist(), past_key_values))
                if past_key_values is None:
                    logits = torch.tensor([[[0.0, 0.0, 5.0, 0.0],
                                            [0.0, 0.0, 5.0, 0.0]]])
                    hidden = torch.zeros(1, 2, 2)
                    cache = "frozen_kv"
                else:
                    assert past_key_values == "frozen_kv"
                    assert input_ids.tolist() == [[2]]
                    logits = torch.tensor([[second_scores]])
                    hidden = torch.zeros(1, 1, 2)
                    cache = "next_kv"
                return SimpleNamespace(
                    logits=logits, hidden_states=[hidden], past_key_values=cache)

        def frozen(model, *args):
            initial = model(
                input_ids=torch.tensor([[7, 8]]),
                past_key_values=None, use_cache=True, output_hidden_states=True)
            model(input_ids=torch.tensor([[2]]),
                  past_key_values=initial.past_key_values,
                  use_cache=True, output_hidden_states=True)
            return [2, token2, 0], {
                "parallel": False, "neural_override_applied": override}

        return FrozenModel(), frozen, calls

    def test_conditional_replay_matches_frozen_cpu_step2_logits(self):
        """Actual original KV-cache forward results are observed, never replayed."""
        import torch
        model, frozen, calls = self._frozen_mock_decoder(
            second_scores=[3.0, 4.0, 1.0, 0.0], token2=1)
        tok = type("Tok", (), {"eos_token_id": 0})()
        traces = {}
        with (patch.object(t, "ORIGINAL_DECODER", frozen),
              patch.object(t.s160, "prompt_tokens", return_value=[7, 8]),
              patch.object(t, "REPLAY_BY_SHA", traces),
              patch.object(t, "CAPTURING", True)):
            result = t._traced_decoder(
                model, None, torch.nn.Identity(), torch.nn.Identity(),
                torch.zeros(4, 2), tok, "prompt", None, None, None)
        self.assertEqual(result[0], [2, 1, 0])
        self.assertEqual(next(iter(traces.values()))["conditional"],
                         [3.0, 4.0, 1.0, 0.0])
        self.assertEqual(calls, [([[7, 8]], None), ([[2]], "frozen_kv")])
        self.assertFalse(model._forward_hooks, "S179 capture hooks must be removed")

    def test_capture_does_not_substitute_scores_for_frozen_decoder(self):
        """Real observed scores may not override original copy-net's choice."""
        import torch
        model, frozen, calls = self._frozen_mock_decoder(
            second_scores=[0.25, 0.5, 1.0, 0.0], token2=1, override=True)
        tok = type("Tok", (), {"eos_token_id": 0})()
        traces = {}
        with (patch.object(t, "ORIGINAL_DECODER", frozen),
              patch.object(t.s160, "prompt_tokens", return_value=[7, 8]),
              patch.object(t, "REPLAY_BY_SHA", traces),
              patch.object(t, "CAPTURING", True)):
            answer, route = t._traced_decoder(
                model, None, torch.nn.Identity(), torch.nn.Identity(),
                torch.zeros(4, 2), tok, "original prompt", None, None, None)
        self.assertEqual(answer, [2, 1, 0])
        self.assertTrue(route["neural_override_applied"])
        self.assertEqual(next(iter(traces.values()))["conditional"],
                         [0.25, 0.5, 1.0, 0.0])
        self.assertEqual(len(calls), 2)

    def test_actual_token2_disagreement_fails_closed(self):
        """Detected true mismatch must fail without extra model forwards."""
        import torch
        model, frozen, calls = self._frozen_mock_decoder(
            second_scores=[0.2, 0.3, 0.9, 0.0], token2=1)
        tok = type("Tok", (), {"eos_token_id": 0})()
        with (patch.object(t, "ORIGINAL_DECODER", frozen),
              patch.object(t.s160, "prompt_tokens", return_value=[7, 8]),
              patch.object(t, "REPLAY_BY_SHA", {}),
              patch.object(t, "CAPTURING", True)):
            with self.assertRaisesRegex(AssertionError, "observer drift"):
                t._traced_decoder(
                    model, None, torch.nn.Identity(), torch.nn.Identity(),
                    torch.zeros(4, 2), tok, "frozen prompt", None, None, None)
        self.assertEqual(len(calls), 2)
        self.assertFalse(model._forward_hooks, "hooks removed even on failure")

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
