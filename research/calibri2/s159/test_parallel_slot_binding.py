"""S159 CPU-only contracts. Synthetic tensors are not real S142 skill gains."""
import unittest
import torch

from calibri2.s159.parallel_slot_binding import (
    ParallelSlotBindingHead, GuardedParallelDecode, source_span_nll,
)


class ParallelBindingContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(159)

    def test_two_tokens_always_come_from_one_contiguous_span(self):
        head = ParallelSlotBindingHead(hidden_size=12, slots=2, rank=4)
        states = torch.randn(3, 10, 12)
        ids = torch.tensor([
            [101, 11, 22, 102, 33, 44, 200, 201, 202, 203],
            [101, 55, 66, 102, 77, 88, 200, 201, 202, 203],
            [101, 19, 20, 102, 31, 32, 200, 201, 202, 203],
        ])
        out = head(states, ids)
        self.assertEqual(tuple(out["slot_ids"].shape), (3, 2))
        self.assertEqual(tuple(out["slot_logits"].shape), (3, 9, 2))
        for b, start in enumerate(out["selected_span"].tolist()):
            self.assertEqual(out["slot_ids"][b].tolist(), ids[b, start:start + 2].tolist())

    def test_parallel_three_slots_not_independent_token_mix(self):
        head = ParallelSlotBindingHead(hidden_size=8, slots=3, rank=5)
        ids = torch.tensor([[4, 5, 6, 9, 8, 7, 42]])
        out = head(torch.randn(1, 7, 8), ids)
        start = int(out["selected_span"].item())
        self.assertEqual(out["slot_ids"][0].tolist(), ids[0, start:start + 3].tolist())
        self.assertEqual(out["slot_logits"].shape[-1], 3)

    def test_mask_excludes_pad_and_invalid_source_tokens(self):
        head = ParallelSlotBindingHead(hidden_size=8, slots=2, rank=3)
        ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
        mask = torch.tensor([[True, True, True, True, False, False]])
        out = head(torch.randn(1, 6, 8), ids, mask)
        self.assertEqual(out["candidate_count"].item(), 3)
        self.assertTrue(torch.isneginf(out["span_logits"][0, 3:]).all())
        self.assertLess(int(out["selected_span"].item()), 3)

    def test_train_only_multi_match_nll_and_both_slot_gradients(self):
        head = ParallelSlotBindingHead(hidden_size=16, slots=2, rank=6)
        states = torch.randn(2, 10, 16)
        ids = torch.tensor([
            [88, 11, 12, 20, 11, 12, 19, 8, 5, 10],
            [88, 10, 11, 20, 11, 12, 19, 8, 5, 10],
        ])
        gold = torch.tensor([[11, 12], [11, 12]])
        opt = torch.optim.Adam(head.parameters(), lr=.02)
        initial = float(source_span_nll(head(states, ids), ids, gold))
        for _ in range(12):
            opt.zero_grad()
            loss = source_span_nll(head(states, ids), ids, gold)
            loss.backward()
            opt.step()
        current = float(source_span_nll(head(states, ids), ids, gold))
        self.assertLess(current, initial)
        self.assertIsNotNone(head.keys.weight.grad)
        self.assertTrue(bool(torch.isfinite(head.keys.weight.grad).all()))
        self.assertGreater(float(head.keys.weight.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            source_span_nll(head(states, ids), ids, torch.tensor([[987, 654], [11, 12]]))

    def test_default_bypass_preserves_parent_exactly(self):
        head = ParallelSlotBindingHead(hidden_size=8, slots=2, rank=3)
        gate = GuardedParallelDecode(head, eos_id=2)
        calls = []
        def parent():
            calls.append(True)
            return torch.tensor([[33, 44, 2]])
        out, used = gate.decode(torch.randn(1, 6, 8), torch.arange(6)[None, :], parent)
        self.assertFalse(used)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(out, torch.tensor([[33, 44, 2]])))
        self.assertFalse(gate.enabled)

    def test_confidence_gate_does_not_call_parent_for_safe_single_span(self):
        head = ParallelSlotBindingHead(hidden_size=8, slots=2, rank=3)
        gate = GuardedParallelDecode(head, eos_id=2, min_confidence=.9, min_margin=2.)
        gate.enabled = True  # ONLY synthetic unit test. Not physical authorization.
        def fake_forward(states, source_ids, source_mask=None):
            return {
                "slot_ids": torch.tensor([[15, 16]]),
                "confidence": torch.tensor([.999]),
                "margin": torch.tensor([5.]),
                "candidate_count": torch.tensor([5]),
            }
        head.forward = fake_forward
        def forbidden():
            raise AssertionError("fast path unnecessarily called original decoding")
        out, used = gate.decode(torch.randn(1, 6, 8), torch.arange(6)[None, :], forbidden)
        self.assertTrue(used)
        self.assertEqual(out.tolist(), [[15, 16, 2]])

    def test_low_confidence_always_falls_back(self):
        head = ParallelSlotBindingHead(hidden_size=8, slots=2, rank=3)
        gate = GuardedParallelDecode(head, eos_id=2, min_confidence=.99)
        gate.enabled = True
        def uncertain(states, source_ids, source_mask=None):
            return {
                "slot_ids": torch.tensor([[15, 16]]),
                "confidence": torch.tensor([.75]),
                "margin": torch.tensor([10.]),
                "candidate_count": torch.tensor([5]),
            }
        head.forward = uncertain
        out, used = gate.decode(torch.randn(1, 6, 8), torch.arange(6)[None, :],
                                lambda: torch.tensor([[21, 22, 2]]))
        self.assertFalse(used)
        self.assertEqual(out.tolist(), [[21, 22, 2]])

    def test_never_assume_multi_batch_speed_guard(self):
        head = ParallelSlotBindingHead(hidden_size=8, slots=2, rank=3)
        gate = GuardedParallelDecode(head, eos_id=2)
        gate.enabled = True
        out, used = gate.decode(torch.randn(2, 6, 8), torch.arange(6)[None, :].repeat(2, 1),
                                lambda: torch.tensor([[21, 22, 2], [31, 32, 2]]))
        self.assertFalse(used)
        self.assertEqual(out.tolist(), [[21, 22, 2], [31, 32, 2]])

    def test_invalid_inputs_fail_closed(self):
        h = ParallelSlotBindingHead(hidden_size=8, slots=2, rank=3)
        with self.assertRaises(ValueError):
            h(torch.randn(2, 1, 8), torch.tensor([[5], [6]]))
        with self.assertRaises(ValueError):
            h(torch.randn(1, 3, 8), torch.tensor([[1, 2, 3]]),
              torch.tensor([[False, False, False]]))


if __name__ == "__main__":
    unittest.main()
