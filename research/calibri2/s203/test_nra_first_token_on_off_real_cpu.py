"""S203 pure contracts; the actual model test must run native CPU J7 in CI."""
from __future__ import annotations
import hashlib
import unittest
from calibri2.s203.nra_first_token_on_off_real_cpu import (
    ORIGINAL_S73_TRAIN_SHA, first_token_observation,
)


class S203RealCPUFirstTokenContract(unittest.TestCase):
    def test_precommitted_only_original_four_s73_train_questions(self):
        known = (
            ("S73/73012/left", "TRAIN S73 ticker: Clerk Ravkin has the two-part inventory key yellow ring. Clerk Yodela has the distinct key yellow drum. Print only the two words of Ravkin's key."),
            ("S73/73012/right", "TRAIN S73 ticker: Clerk Ravkin has the two-part inventory key yellow drum. Clerk Yodela has the distinct key yellow ring. Print only the two words of Ravkin's key."),
            ("S73/73015/left", "TRAIN S73 label: The assignment is Ceiran : yellow ring; also Elnov : yellow drum. State exactly the two assigned words for Ceiran in order."),
            ("S73/73015/right", "TRAIN S73 label: The assignment is Ceiran : yellow drum; also Elnov : yellow ring. State exactly the two assigned words for Ceiran in order."),
        )
        self.assertEqual(len(ORIGINAL_S73_TRAIN_SHA), 4)
        for identity, q in known:
            self.assertEqual(ORIGINAL_S73_TRAIN_SHA[hashlib.sha256(q.encode()).hexdigest()], identity)

    def test_neural_first_token_ablation_detects_shift_not_gold(self):
        observed = first_token_observation([16591, 8233, 2], [23429, 8233, 2])
        self.assertTrue(observed["first_token_changed_by_NRA_activation"])
        self.assertEqual(observed["NRA_ON_first_token"], 16591)
        self.assertEqual(observed["NRA_OFF_first_token"], 23429)
        self.assertTrue(observed["full_generated_sequence_changed"])
        self.assertNotIn("expected", str(observed).lower())

    def test_no_first_token_shift_is_not_false_gain(self):
        observed = first_token_observation([23429, 8233, 2], [23429, 8233, 2])
        self.assertFalse(observed["first_token_changed_by_NRA_activation"])
        self.assertFalse(observed["full_generated_sequence_changed"])
        for ids in ([], [2]):
            if ids == []:
                with self.assertRaises(AssertionError):
                    first_token_observation(ids, [23429, 2])


if __name__ == "__main__":
    unittest.main()
