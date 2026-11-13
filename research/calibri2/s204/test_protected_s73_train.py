"""S204 signed original S73 TRAIN source checks (unit-only, not CPU J7)."""
import unittest

from calibri2.s204.protected_s73_train import (
    ORIGINAL_S73_SHA,
    precommitted_s73_original_train,
)


class S204OriginalS73SourceTests(unittest.TestCase):
    @staticmethod
    def old_original():
        # The four prompts are the literal S202 posthoc original TRAIN
        # transcripts, never generated heldout QA/TEST samples. The 452
        # placeholders are unit-test structural fixtures only: no CPU decode.
        samples = {f"TEST_FIXTURE_{i}": {} for i in range(452)}
        samples.update({
            "S73/73012/left": {
                "question": "TRAIN S73 ticker: Clerk Ravkin has the two-part inventory key yellow ring. Clerk Yodela has the distinct key yellow drum. Print only the two words of Ravkin's key.",
                "expected": [23429, 8233, 2],
            },
            "S73/73012/right": {
                "question": "TRAIN S73 ticker: Clerk Ravkin has the two-part inventory key yellow drum. Clerk Yodela has the distinct key yellow ring. Print only the two words of Ravkin's key.",
                "expected": [23429, 15754, 2],
            },
            "S73/73015/left": {
                "question": "TRAIN S73 label: The assignment is Ceiran : yellow ring; also Elnov : yellow drum. State exactly the two assigned words for Ceiran in order.",
                "expected": [23429, 8233, 2],
            },
            "S73/73015/right": {
                "question": "TRAIN S73 label: The assignment is Ceiran : yellow drum; also Elnov : yellow ring. State exactly the two assigned words for Ceiran in order.",
                "expected": [23429, 15754, 2],
            },
        })
        return samples

    def test_exact_original_s73_train_commitments(self):
        rows = precommitted_s73_original_train(self.old_original())
        self.assertEqual(len(rows), 4)
        self.assertEqual(tuple(id for id, _ in rows),
                         tuple(ORIGINAL_S73_SHA))
        self.assertEqual(tuple(row["expected"][0] for _id, row in rows),
                         (23429,) * 4)

    def test_missing_or_mutated_train_question_rejected(self):
        old = self.old_original()
        del old["S73/73015/left"]
        with self.assertRaises(AssertionError):
            precommitted_s73_original_train(old)
        old = self.old_original()
        old["S73/73015/left"]["question"] += " "
        with self.assertRaises(AssertionError):
            precommitted_s73_original_train(old)

    def test_missing_eos_or_wrong_total_rejected(self):
        old = self.old_original()
        old["S73/73015/right"]["expected"][-1] = 3
        with self.assertRaises(AssertionError):
            precommitted_s73_original_train(old)
        old = self.old_original()
        del old["TEST_FIXTURE_2"]
        with self.assertRaises(AssertionError):
            precommitted_s73_original_train(old)


if __name__ == "__main__":
    unittest.main()
