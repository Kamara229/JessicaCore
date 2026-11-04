"""Small deterministic protocol tests; no Jessica/live integration."""
import unittest
import re

import torch
from v130.protocol import MAPPINGS, cases, shuffled_questions
from v130.metrics import grade, similarity, swap_alignment
from v130.capture import compare, POINTS


class V130Tests(unittest.TestCase):
    def test_reversal_uses_the_same_labels_and_both_foci(self):
        rows = cases()
        self.assertEqual(len(rows), 4 * len(MAPPINGS))
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(len(by_id), len(rows))
        for key, *_ in MAPPINGS:
            for focus in ("left", "right"):
                original = by_id[f"{key}:{focus}:original"]
                reversed_ = by_id[f"{key}:{focus}:swapped"]
                self.assertEqual(original["gold"], reversed_["foil"])
                self.assertEqual(original["foil"], reversed_["gold"])
                self.assertNotEqual(original["question"], reversed_["question"])
                self.assertEqual(sorted(re.findall(r"[A-Za-z]+", original["question"].casefold())),
                                 sorted(re.findall(r"[A-Za-z]+", reversed_["question"].casefold())))

    def test_shuffle_has_no_self_question_and_preserves_record_count(self):
        rows = cases()
        other = shuffled_questions(rows)
        self.assertEqual(len(other), len(rows))
        self.assertTrue(all(other[r["id"]] != r["question"] for r in rows))
        self.assertEqual({r["question"] for r in rows}, set(other.values()))

    def test_do_not_confuse_first_word_with_full_answer(self):
        sample = grade("amber cobalt", "amber", False)
        self.assertTrue(sample["first_word_matches"])
        self.assertFalse(sample["exact_text"])
        self.assertFalse(sample["exact_text_with_eos"])
        self.assertFalse(grade("silver", "amber", True)["first_word_matches"])
        self.assertTrue(grade("amber.", "amber", True)["exact_text_with_eos"])

    def test_directional_sign_and_error(self):
        a = torch.tensor([1., 0.])
        b = torch.tensor([0., 1.])
        self.assertAlmostEqual(similarity(a, a)["cosine"], 1.)
        self.assertAlmostEqual(similarity(a, b)["cosine"], 0.)
        diag = swap_alignment(a, b, a, b)
        self.assertAlmostEqual(diag["cosine"], 1.)
        flipped = swap_alignment(a, b, b, a)
        self.assertAlmostEqual(flipped["cosine"], -1.)
        self.assertIsNone(swap_alignment(a, a, b, a))

    def test_same_activations_have_no_spurious_drift(self):
        rows = cases()
        exemplar = {p: torch.tensor([1., 2., 3.]) for p in POINTS}
        refs = {r["id"]: exemplar for r in rows}
        scores = compare(refs, refs, rows)
        self.assertEqual(set(scores), set(POINTS))
        for point in POINTS:
            self.assertAlmostEqual(scores[point]["mean_relative_hidden_error"], 0.)
            self.assertEqual(scores[point]["swap_count_with_nonzero_teacher_signal"], 0)


if __name__ == "__main__":
    unittest.main()
