"""Contract tests for pinned archival JSON replay; stdlib only; no 43M model."""
import copy
import unittest
from research.calibri_nra3.s119q2i8z39_spent14_native_diversity_forensics import (
    STAGE, DEV_SHA, BASE, TRUE, WRONG, MODES, audit, empirical_diversity, paired_sign_p
)


def synthetic_fixture():
    rows = []
    for i in range(14):
        gold = [7, 8, 9, 10]
        base2 = 8 if i == 0 else 100 + (i % 10)
        chosen2 = {BASE: base2,
                   TRUE: 8 if i < 4 else 30 + (i % 3),
                   WRONG: 50 + (i % 6)}
        candidates = {}
        for mode in MODES:
            second = chosen2[mode]
            seq = [7, second, 70 + i, 80 + i, 90 + i, 91 + i]
            steps = []
            for position in range(2, 7):
                chosen = seq[position - 1]
                steps.append({
                    "native_position_1based": position,
                    "chosen_native_id": chosen,
                    "original_greedy_native_id": base2 if position == 2 else chosen,
                    "candidate_original_top64": True,
                    "shadow_changed_original_greedy": second != base2 if position == 2 else False
                })
            candidates[mode] = {
                "autonomous_first_six_native_ids": seq,
                "first_native_correct": True,
                "first_two_native_correct": second == 8,
                "first_three_native_correct": False,
                "per_second_to_fourth_real_forward": steps
            }
        rows.append({
            "source_id": "TEST_ONLY_" + str(i),
            "human_gold_native_first4_POSTHOC": gold,
            "DEV_gold_was_never_in_training_or_candidate_selection": True,
            "auto_generated_modes_no_teacher_forcing": candidates
        })
    scores = {}
    for mode in MODES:
        second = [r["auto_generated_modes_no_teacher_forcing"][mode]["autonomous_first_six_native_ids"][1] for r in rows]
        scores[mode] = {
            "correct_FIRST_TWO": sum(x == 8 for x in second),
            "correct_FIRST_THREE": 0,
            "distinct_second_native_BPE_ids": len(set(second))
        }
    arm = {
        "NEW_DEV14_per_prompt_first6": rows,
        "NEW_DEV14_full_autonomous_first6_native": {
            "matched_autonomous_first6_native_generated": scores
        }
    }
    report = {
        "stage": STAGE,
        "Z37_PRECOMMITTED_14_BRANDNEW_DEV_SHA256": DEV_SHA,
        "Z37_brandNEW_author_DEV14_NOW_SPENT_not_independently_blind": True,
        "output_gate": 0, "candidate_accepted": False,
        "original43M_49k_native_FORWARDS": 752,
        "actual_original43M_A_B_real_native49k_BPE2_only_rank32":
            {"TEST_A": arm, "TEST_B": copy.deepcopy(arm)},
    }
    for name in (
        "strict_ONLY_second_native_BPE_candidate_decoding_intervention",
        "original_S142_43M_GT12_Core_NRA3_NPA1_and_Z10_A_B_unchanged",
        "no_gold_in_candidate_selection_on_new_DEV",
        "no_previous_SPENT_DEV_used_in_TRAIN",
        "no_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification",
    ):
        report[name] = True
    return report


class TestZ39Replay(unittest.TestCase):
    def test_empirical_diversity_is_cross_prompt_not_predictive_entropy(self):
        self.assertEqual(empirical_diversity([1, 1, 1, 1])["empirical_entropy_bits"], 0)
        self.assertEqual(empirical_diversity([1, 2, 3, 4])["empirical_entropy_bits"], 2)
    def test_exact_small_sample_p_is_not_overstated(self):
        self.assertEqual(paired_sign_p(3, 0), .25)
        self.assertEqual(paired_sign_p(0, 0), 1)
    def test_archived_rank2_improvement_and_rank3_bottleneck(self):
        result = audit(synthetic_fixture())
        arm = result["observed_arms"]["TEST_A"]
        self.assertEqual((arm[BASE]["first2"], arm[TRUE]["first2"],
                          arm[WRONG]["first2"]), (1, 4, 0))
        self.assertEqual((arm[BASE]["first3"], arm[TRUE]["first3"]), (0, 0))
        self.assertEqual(arm[TRUE]["first2_correct_but_third_wrong"], 4)
        self.assertEqual(arm[TRUE]["paired_BPE2_newly_correct"], 3)
        self.assertEqual(arm[TRUE]["exact_discordant_two_sided_p_descriptive"], .25)
        self.assertEqual((arm[BASE]["distinct_ids"], arm[TRUE]["distinct_ids"]), (11, 4))
        self.assertFalse(result["candidate_accepted"])
        self.assertFalse(result["validation_or_replication"])
    def test_fail_closed_on_leak_or_unexpected_decoder_rewrite(self):
        sample = synthetic_fixture()
        sample["no_gold_in_candidate_selection_on_new_DEV"] = False
        with self.assertRaises(ValueError): audit(sample)
        sample = synthetic_fixture()
        row = sample["actual_original43M_A_B_real_native49k_BPE2_only_rank32"]["TEST_A"]["NEW_DEV14_per_prompt_first6"][0]
        row["auto_generated_modes_no_teacher_forcing"][TRUE]["per_second_to_fourth_real_forward"][1]["original_greedy_native_id"] = 999
        with self.assertRaises(ValueError): audit(sample)
    def test_fail_closed_if_archive_changed_or_unvetted(self):
        sample = synthetic_fixture()
        sample["original43M_49k_native_FORWARDS"] = 751
        with self.assertRaises(ValueError): audit(sample)
        sample = synthetic_fixture()
        sample["Z37_PRECOMMITTED_14_BRANDNEW_DEV_SHA256"] = "bad"
        with self.assertRaises(ValueError): audit(sample)


if __name__ == "__main__":
    unittest.main()
