"""Offline contract tests: NO model weights, QA/TEST or backend calls."""
import inspect
import unittest

import torch

from calibri2.s177.adaptive_consensus import adaptive_consensus


class AdaptiveConsensusContracts(unittest.TestCase):
    def setUp(self):
        self.spans = torch.tensor([[11, 22], [33, 44], [11, 22]], dtype=torch.long)
        self.logits = torch.tensor([9.0, 0.0, 7.0])
        self.args = dict(question="TRAIN S82: Al has blue book. Give Al's key.",
                         eos_id=2, legacy_choice=(33, 44))

    def choose(self, **overrides):
        data = dict(self.args, verified_owner_ids=(11, 22))
        data.update(overrides)
        return adaptive_consensus(self.logits, self.spans, **data)

    def test_promotes_only_independently_verified_whole_pair(self):
        result = self.choose()
        self.assertEqual(result.proposed, (11, 22))
        self.assertTrue(result.changes_legacy)
        self.assertEqual(result.reason, "verified_consensus_5")
        self.assertEqual(result.unique_span_count, 2)
        self.assertEqual(result.branch_count, 5)

    def test_no_verifier_no_promotion(self):
        result = self.choose(verified_owner_ids=None)
        self.assertIsNone(result.proposed)
        self.assertFalse(result.changes_legacy)
        self.assertEqual(result.reason, "no_independent_owner_proof")
        self.assertEqual(result.branch_count, 0)

    def test_wrong_owner_never_accepted_on_popularity_alone(self):
        result = self.choose(verified_owner_ids=(33, 44))
        self.assertIsNone(result.proposed)
        self.assertEqual(result.reason, "no_verified_consensus")

    def test_no_cross_owner_token_recombination(self):
        result = self.choose(verified_owner_ids=(11, 44))
        self.assertIsNone(result.proposed)
        self.assertEqual(result.reason, "owner_not_exact_source_span")

    def test_negative_and_eos_owner_rejected(self):
        for value in ((-1, 22), (11, 2), (11, 22, 33)):
            result = self.choose(verified_owner_ids=value)
            self.assertIsNone(result.proposed)

    def test_repeated_spans_not_extra_votes_and_empty_pool(self):
        result = self.choose()
        self.assertEqual(result.unique_span_count, 2)
        collapsed = adaptive_consensus(
            torch.tensor([10.0, 9.0]), torch.tensor([[11, 22], [11, 22]]),
            **{**self.args, "verified_owner_ids": (11, 22)})
        self.assertEqual(collapsed.reason, "insufficient_distinct_candidates")

    def test_local_rng_does_not_modify_global_torch_rng(self):
        torch.manual_seed(98765)
        before = torch.random.get_rng_state().clone()
        self.choose()
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(self.choose(), self.choose())

    def test_5_or_10_branches_only_and_ten_path_possible(self):
        weak = torch.tensor([0.8, 0.0, -0.5])
        spans = torch.tensor([[11, 22], [33, 44], [55, 66]])
        observed_ten = False
        for number in range(500):
            result = adaptive_consensus(
                weak, spans, question="TRAIN example "+str(number),
                verified_owner_ids=(11, 22), legacy_choice=(33, 44), eos_id=2)
            self.assertIn(result.branch_count, (5, 10))
            if result.reason == "verified_consensus_10":
                observed_ten = True
                break
        self.assertTrue(observed_ten)

    def test_only_own_prompt_and_logits_no_gold_api(self):
        signature = str(inspect.signature(adaptive_consensus))
        self.assertNotIn("gold", signature.lower())
        self.assertNotIn("expected", signature.lower())
        with self.assertRaises(ValueError):
            adaptive_consensus(torch.zeros(2), torch.ones((2, 3), dtype=torch.long),
                               **{**self.args, "verified_owner_ids": (11, 22)})

    def test_unverified_or_nonfinite_source_spans_are_discarded(self):
        logits = torch.tensor([float("nan"), 4.0, 3.0])
        spans = torch.tensor([[11, 22], [33, 44], [-1, 66]])
        result = adaptive_consensus(
            logits, spans, **{**self.args, "verified_owner_ids": (11, 22)})
        self.assertIsNone(result.proposed)


if __name__ == "__main__":
    unittest.main()
