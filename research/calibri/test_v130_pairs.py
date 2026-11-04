"""Fast static protocol and grading regression tests. No model invocation."""
import unittest
from v130.protocol import PAIRS, tasks, validate, words
from v130.scoring import evidence, summarize
from v130.analysis import compare, ARMS


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

    def test_teacher_conditioned_swapped_green_control(self):
        arms = {}
        for arm in ARMS:
            rows = []
            for pair in PAIRS:
                for side, question, gold, foil in (
                    ("left", pair.left, pair.left_answer, pair.right_answer),
                    ("right", pair.right, pair.right_answer, pair.left_answer),
                ):
                    # Teacher has one valid pair only. Current Green fixes it.
                    value = gold if (arm == "FP32" and pair == PAIRS[0]) or (
                        arm == "v119_green_current" and pair == PAIRS[0]) else foil
                    rows.append({"pair": pair.name, "side": side,
                                 "question": question,
                                 "reference_for_post_generation_grading_only": gold,
                                 "generated_token_ids": [ord(value[0])],
                                 "observation": evidence(value, question, gold, foil, True)})
            arms[arm] = {"rows": rows}
        report = compare(arms)
        result = report["metrics"]["strict_exact_text"]
        self.assertEqual(result["fp32_pair_eligible"], 1)
        self.assertEqual(result["eligible_pairs"]["v119_green_current"], 1)
        self.assertEqual(result["green_current_unique_wins_vs_both_controls"], 1)
        arms["CB59"]["rows"][0]["question"] = "DIFFERENT"
        with self.assertRaises(ValueError):
            compare(arms)

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
