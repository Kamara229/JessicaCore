"""Fast static protocol and grading regression tests. No model invocation."""
import unittest
from v130.protocol import PAIRS, tasks, validate, words
from v130.scoring import evidence, summarize


class TestCounterfactualPairs(unittest.TestCase):
    def test_pairs_swap_relations_not_words(self):
        validate()
        self.assertEqual(len(PAIRS), 8)
        for p in PAIRS:
            self.assertEqual(words(p.left), words(p.right))
            self.assertNotEqual(p.left_answer, p.right_answer)
        self.assertEqual(len(list(tasks())), 16)

    def test_cannot_count_both_answers_as_right(self):
        e = evidence("Mira has blue and Niko has red.", "", "blue", "red", False)
        self.assertTrue(e["both_alternatives_visible"])
        self.assertFalse(e["target_visible_alone_NOT_proven_semantic"])

    def test_exact_and_eos_are_different(self):
        e = evidence("Blue.", "Which color?", "blue", "red", False)
        self.assertTrue(e["strict_exact_text"])
        self.assertFalse(e["strict_exact_with_eos"])

    def test_negative_and_no_content(self):
        self.assertTrue(evidence("red", "", "blue", "red", True)["foil_visible_alone"])
        self.assertTrue(evidence("", "", "blue", "red", True)["no_alternative_visible"])

    def test_both_sides_must_pass_for_pair(self):
        rows=[]
        for p in PAIRS:
            for side, gold, foil in (("left",p.left_answer,p.right_answer),
                                      ("right",p.right_answer,p.left_answer)):
                value = gold if not (p.name=="key_color" and side=="right") else foil
                rows.append({"pair":p.name,"side":side,"observation":evidence(
                    value, "", gold, foil, True)})
        counts=summarize(rows)
        self.assertEqual(counts["number_of_counterfactual_pairs"],8)
        self.assertEqual(counts["both_sides_strict_exact"],7)
        self.assertEqual(counts["both_sides_unique_target_visible"],7)


if __name__ == "__main__":
    unittest.main()
