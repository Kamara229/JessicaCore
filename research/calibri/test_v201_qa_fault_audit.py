"""CALIBRI v201: synthetic contract tests only; no Jessica/model inference."""
import copy
import unittest
from v201.audit import analyze


def sample(arm, i, text, status, tokens=None, eos=False, cap=False):
    return {
        "case_id": i,
        "verbatim_text": text,
        "token_ids": tokens if tokens is not None else [4, 5, 6],
        "eos": eos, "cap_hit_without_eos": cap,
    }


def fixture():
    cases = [{"case_id": i, "domain": "arithmetic",
              "question": "What is one plus one?", "gold": "2"} for i in range(16)]
    scored = {}
    original = [{"case_id": i, "verbatim_text": "2" if i < 6 else "?"
                } for i in range(16)]
    compact = {"CB186": [], "GT200": []}
    for arm, count in (("FP32_existing_135M", 6), ("CB186", 2), ("GT200", 3)):
        grades = []
        for i in range(16):
            correct = i < count
            grades.append({"case_id": i, "raw": "2" if correct else "?",
                           "machine_status": "machine_correct" if correct else "ungraded",
                           "candidate": "2" if correct else None,
                           "recognition_basis": "synthetic"})
            if arm != "FP32_existing_135M":
                compact[arm].append({
                    "case_id": i, "actual_output": sample(
                        arm, i, "2" if correct else "?",
                        "machine_correct" if correct else "ungraded",
                        [2, 100] if correct else [8]*9,
                        eos=correct, cap=not correct)})
        scored[arm] = {
            "n":16, "machine_confirmed":count, "machine_recognized_wrong":0,
            "ungraded_not_known_wrong":16-count,
            "verbatim_answers_and_grades": grades,
        }
    return {
        "complete": True, "not_new_blind_test_public_dev_only": True,
        "score_changes_not_used_to_tune_the_training_that_just_ran": True,
        "private_v89_TEST_unread": True,
        "cases": cases,
        "original_actual_135M_FP32_from_previous_sealed_v193": original,
        "native_compact_actual_QA": compact,
        "scores": scored,
    }


class ReadOnlyQAContract(unittest.TestCase):
    def test_alignment_parity_and_repetition(self):
        report = analyze(fixture())
        self.assertEqual(report["machine_correct"]["GT200"],3)
        self.assertEqual(report["failure_taxonomy"]["repeated_token_loop"],13)
        self.assertEqual(report["failure_taxonomy"]["confirmed_FP32_correct_GT200_not_confirmed"],3)
        self.assertEqual(report["failure_taxonomy"]["became_confirmed_correct"],1)
        self.assertTrue(report["read_only_reanalysis_of_sealed_v200_artifact"])
        self.assertFalse(report["source_changed"])

    def test_mismatched_gold_response_rejected(self):
        data=fixture()
        data["native_compact_actual_QA"]["GT200"][0]["actual_output"]["verbatim_text"]="different"
        with self.assertRaises(AssertionError):
            analyze(data)

    def test_mismatched_grades_rejected(self):
        data=fixture()
        data["scores"]["GT200"]["machine_confirmed"]=4
        with self.assertRaises(AssertionError):
            analyze(data)

    def test_no_unsealed_or_missing_arm(self):
        data=fixture()
        data["not_new_blind_test_public_dev_only"]=False
        with self.assertRaises(AssertionError):
            analyze(data)
        data=fixture()
        del data["native_compact_actual_QA"]["GT200"][0]
        with self.assertRaises(AssertionError):
            analyze(data)


if __name__ == "__main__":
    unittest.main()
